#!/usr/bin/env python3
"""
Analyze a fit-cache binary file and print its structure.
Usage: python3 fit-cache-analyze.py <cache-file.bin>
"""

import struct
import sys

def read_u32_le(data, offset):
    return struct.unpack_from('<I', data, offset)[0]

def read_u64_le(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]

def read_i64_le(data, offset):
    return struct.unpack_from('<q', data, offset)[0]

def analyze_cache(path):
    with open(path, 'rb') as f:
        data = bytearray(f.read())

    print(f"File: {path}")
    print(f"Size: {len(data)} bytes")
    print()

    if len(data) < 28:
        print("File too small to contain header")
        return

    magic = bytes(data[:4])
    print(f"Magic: {magic} ({'OK' if magic == b'CTIF' else 'BAD'})")
    version = read_u32_le(data, 4)
    print(f"Version: {version}")
    key_hash = read_u64_le(data, 8)
    print(f"Key hash: {key_hash:016x}")
    timestamp = read_u64_le(data, 16)
    print(f"Timestamp: {timestamp}")
    body_size = read_u32_le(data, 24)
    print(f"Body size: {body_size} bytes")
    print(f"Expected total: {28 + body_size} bytes, actual: {len(data)} bytes")
    print()

    if magic != b'CTIF':
        print("Not a CTIF file")
        return

    if len(data) < 28 + body_size:
        print(f"File truncated: expected {28 + body_size} bytes, got {len(data)}")
        body_size = len(data) - 28

    # Walk body
    pos = 28
    n_roles = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] n_roles: {n_roles}")

    # Role devices
    print(f"[{pos:4d}] role_devices: [{n_roles} device ids]")
    for i in range(n_roles):
        v = read_u32_le(data, pos + 4*i)
        print(f"  device_id[{i}] = {v}")
    pos += 4 * n_roles

    n_gpu_layers = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] n_gpu_layers: {n_gpu_layers}")

    ts_count = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] tensor_split count: {ts_count}")
    for i in range(min(ts_count, 16)):
        v = read_u32_le(data, pos + 4*i)
        print(f"  tensor_split[{i}] = {v}")
    if ts_count > 16:
        print(f"  ... ({ts_count - 16} more)")
    pos += 4 * ts_count

    n_ctx = read_u64_le(data, pos)
    pos += 8
    print(f"[{pos-8:4d}] n_ctx: {n_ctx}")
    pool_grant = read_u64_le(data, pos)
    pos += 8
    print(f"[{pos-8:4d}] pool_grant_mib: {pool_grant}")

    shed_count = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] shed_reserves_count: {shed_count}")
    shed_start = pos
    for i in range(min(shed_count, 8)):
        taglen = read_u32_le(data, pos)
        pos += 4
        tag = data[pos:pos+taglen].decode('utf-8', errors='replace')
        pos += taglen
        print(f"  shed[{i}]: \"{tag}\" (len={taglen})")
    if shed_count > 8:
        print(f"  ... ({shed_count - 8} more)")
    pos = shed_start + sum(read_u32_le(data, shed_start + 4*i) + 4 for i in range(shed_count))

    delta_count = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] delta_free_ema count: {delta_count}")
    delta_values = []
    for i in range(delta_count):
        v = read_i64_le(data, pos + 8*i)
        delta_values.append(v)
        print(f"  delta_free_ema[{i}] = {v} ({v/1024/1024:.2f} MiB)")
    pos += 8 * delta_count

    proj_count = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] projected_free count: {proj_count}")
    for i in range(proj_count):
        v = read_i64_le(data, pos + 8*i)
        print(f"  projected_free[{i}] = {v} ({v/1024/1024:.2f} MiB)")
    pos += 8 * proj_count

    proj_with_delta = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] projected_with_delta: {proj_with_delta}")

    ceil_count = read_u32_le(data, pos)
    pos += 4
    print(f"[{pos-4:4d}] ceiling_free count: {ceil_count}")
    for i in range(ceil_count):
        v = read_i64_le(data, pos + 8*i)
        print(f"  ceiling_free[{i}] = {v} ({v/1024:.0f} MiB)")
    pos += 8 * ceil_count

    print()
    print(f"Body end: pos={pos}, expected={28+body_size}")
    print(f"Remaining bytes after body: {len(data) - (28 + body_size)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 fit-cache-analyze.py <cache-file.bin>")
        sys.exit(1)
    analyze_cache(sys.argv[1])
