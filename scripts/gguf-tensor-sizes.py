#!/usr/bin/env python3
"""
Print the byte total of a GGUF file's tensors, grouped by type.

Why this exists: the TQ types have a load-time conversion that changes how
much device memory a weight tensor occupies. TQ4_1S is 20 bytes per 32 values;
the q8_0 it converts into is 34 bytes per 32 values, so 1.7x. That makes it
possible to tell from the outside whether a device holds the native bytes or
the converted copy - sum the model buffers the loader reports for every
device and compare against this number.

It needs no numpy, so it runs anywhere the build host does.

Usage:
    python3 scripts/gguf-tensor-sizes.py model.gguf
    python3 scripts/gguf-tensor-sizes.py model.gguf --largest
"""
import argparse
import os
import re
import struct
import sys
from collections import Counter, defaultdict

# ggml_type id -> (block size, bytes per block)
# Only the types that can appear in a TQ-era model are listed; anything else
# falls back to 4 bytes per element, which is wrong for exotic quant types.
TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
    35: (256, 66), 39: (32, 17), 40: (16, 8), 41: (256, 54), 42: (256, 66),
    43: (16, 6), 44: (32, 14), 45: (32, 16), 46: (32, 20), 47: (32, 20),
}
NAMES = {
    0: "f32", 1: "f16", 8: "q8_0", 14: "q6_K", 26: "i32", 30: "bf16",
    45: "tq3_1s", 46: "tq4_1s", 43: "turbo2", 44: "turbo3", 47: "turbo4",
}
TQ4_1S = 46


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def u32(self):
        return struct.unpack("<I", self.fh.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.fh.read(8))[0]

    def i64(self):
        return struct.unpack("<q", self.fh.read(8))[0]

    def string(self):
        return self.fh.read(self.u64()).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype == 0:
            return struct.unpack("<B", self.fh.read(1))[0]
        if vtype == 1:
            return struct.unpack("<b", self.fh.read(1))[0]
        if vtype == 2:
            return struct.unpack("<H", self.fh.read(2))[0]
        if vtype == 3:
            return struct.unpack("<h", self.fh.read(2))[0]
        if vtype == 4:
            return self.u32()
        if vtype == 5:
            return struct.unpack("<i", self.fh.read(4))[0]
        if vtype == 6:
            return struct.unpack("<f", self.fh.read(4))[0]
        if vtype == 7:
            return bool(self.fh.read(1)[0])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            elem_type = self.u32()
            return [self.value(elem_type) for _ in range(self.u64())]
        if vtype == 10:
            return self.u64()
        if vtype == 11:
            return self.i64()
        if vtype == 12:
            return struct.unpack("<d", self.fh.read(8))[0]
        raise ValueError("unknown gguf value type %d" % vtype)


def read(path):
    """Return (gguf version, [(name, type id, nbytes, dims), ...])."""
    with open(path, "rb") as fh:
        r = Reader(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            r.string()
            r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            ttype = r.u32()
            r.u64()  # data offset
            n_elem = 1
            for d in dims:
                n_elem *= d
            blck, tsize = TYPES.get(ttype, (1, 4))
            tensors.append((name, ttype, n_elem // blck * tsize, dims))
        return version, tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--largest", action="store_true", help="list the largest tensors")
    args = ap.parse_args()

    version, tensors = read(args.model)
    by_type = Counter()
    by_layer = defaultdict(int)
    for name, ttype, nbytes, _ in tensors:
        by_type[ttype] += nbytes
        m = re.match(r"blk\.(\d+)\.", name)
        by_layer["blk.%s" % m.group(1) if m else name] += nbytes

    total = sum(by_type.values())
    print("%s" % args.model)
    print("  gguf version %d, %d tensors" % (version, len(tensors)))
    for ttype, nbytes in by_type.most_common():
        print("    %-10s (id %2d) %10.2f MiB" % (NAMES.get(ttype, "type_%d" % ttype), ttype, nbytes / 1048576))
    print("    %-10s            %10.2f MiB" % ("TOTAL", total / 1048576))
    print("    %-10s            %10.2f MiB" % ("file size", os.path.getsize(args.model) / 1048576))

    tq4 = by_type.get(TQ4_1S, 0)
    if tq4:
        print()
        print("  TQ4_1S %d bytes native, %d bytes if converted to q8_0 (x1.7)"
              % (tq4, tq4 * 34 // 20))

    if args.largest:
        print()
        print("  largest tensors:")
        for name, ttype, nbytes, dims in sorted(tensors, key=lambda t: -t[2])[:15]:
            print("    %9.2f MiB  %-8s %s" % (nbytes / 1048576, NAMES.get(ttype, "type_%d" % ttype), name[:64]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
