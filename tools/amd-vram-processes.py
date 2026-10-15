#!/usr/bin/env python3
"""Show per-process AMDGPU VRAM and GTT allocations on Linux.

Run with sudo to read the DRM debugfs accounting:
    sudo python3 tools/amd-vram-processes.py
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


DRM_DEVICE_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
CLIENT_RE = re.compile(r"^pid\s+(\d+)\s+command\s+(.+?):\s*$")
OBJECT_RE = re.compile(r"^\s*0x[0-9a-fA-F]+:\s*(\d+)\s+byte\s+(VRAM|GTT|CPU|NONE)\b(.*)$")
EXPORT_RE = re.compile(r"\bexported as ino:(\d+)")


def parse_gem_info(text):
    clients = {}
    current = None
    for line in text.splitlines():
        match = CLIENT_RE.match(line)
        if match:
            pid = int(match.group(1))
            name = match.group(2).strip()
            current = clients.setdefault(pid, {
                "pid": pid,
                "name": name,
                "vram": 0,
                "gtt": 0,
                "shared_objects": set(),
            })
            continue

        if current is None:
            continue
        match = OBJECT_RE.match(line)
        if not match:
            continue

        size = int(match.group(1))
        domain = match.group(2)
        export = EXPORT_RE.search(match.group(3))
        if export:
            inode = export.group(1)
            if inode in current["shared_objects"]:
                continue
            current["shared_objects"].add(inode)
        if domain == "VRAM":
            current["vram"] += size
        elif domain == "GTT":
            current["gtt"] += size

    return clients


def parse_kfd_processes(text):
    processes = {}
    for device in json.loads(text):
        for item in device.get("process_list", []):
            info = item.get("process_info", {})
            try:
                pid = int(info["pid"])
                vram = int(info.get("mem_usage", {}).get("value", 0))
            except (KeyError, TypeError, ValueError):
                continue
            name = Path(str(info.get("name", "unknown"))).name
            current = processes.get(pid)
            if current is None or vram > current["kfd_vram"]:
                processes[pid] = {"pid": pid, "name": name, "kfd_vram": vram}
    return processes


def read_kfd_processes(bdf):
    executable = shutil.which("amd-smi")
    if executable is None:
        return {}, "amd-smi not found"
    try:
        result = subprocess.run(
            [executable, "process", "--general", "--gpu", bdf, "--json"],
            check=True, capture_output=True, text=True,
        )
        return parse_kfd_processes(result.stdout), None
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return {}, f"amd-smi process query failed: {error}"


def discover_devices():
    devices = []
    drm_root = Path("/sys/class/drm")
    for card in drm_root.glob("card[0-9]*"):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            bdf = device.resolve().name
        except OSError:
            continue
        if vendor != "0x1002" or not DRM_DEVICE_RE.fullmatch(bdf):
            continue
        devices.append(bdf.lower())
    return sorted(set(devices))


def resolve_device(value):
    if DRM_DEVICE_RE.fullmatch(value):
        return value.lower()
    card = Path("/sys/class/drm") / value / "device"
    if not value.startswith("card") or not card.exists():
        raise ValueError(f"not a PCI BDF or DRM card name: {value}")
    bdf = card.resolve().name.lower()
    if not DRM_DEVICE_RE.fullmatch(bdf):
        raise ValueError(f"could not resolve a PCI BDF for {value}")
    return bdf


def read_bytes(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def print_device(bdf, input_path=None, kfd_input_path=None):
    debugfs = Path("/sys/kernel/debug/dri") / bdf / "amdgpu_gem_info"
    if input_path:
        gem_info = Path(input_path).read_text()
    else:
        gem_info = debugfs.read_text()

    clients = parse_gem_info(gem_info)
    kfd_warning = None
    if kfd_input_path:
        kfd_processes = parse_kfd_processes(Path(kfd_input_path).read_text())
    elif input_path:
        kfd_processes = {}
    else:
        kfd_processes, kfd_warning = read_kfd_processes(bdf)
    for pid, process in kfd_processes.items():
        client = clients.setdefault(pid, {
            "pid": pid,
            "name": process["name"],
            "vram": 0,
            "gtt": 0,
            "shared_objects": set(),
        })
        client["kfd_vram"] = process["kfd_vram"]

    clients = sorted(clients.values(), key=lambda client: max(
        client.get("kfd_vram", 0), client["vram"] + client["gtt"]), reverse=True)
    device_sysfs = Path("/sys/bus/pci/devices") / bdf
    vram_used = read_bytes(device_sysfs / "mem_info_vram_used")
    vram_total = read_bytes(device_sysfs / "mem_info_vram_total")
    gtt_used = read_bytes(device_sysfs / "mem_info_gtt_used")
    gtt_total = read_bytes(device_sysfs / "mem_info_gtt_total")

    print(f"AMDGPU {bdf}")
    if vram_used is not None and vram_total is not None:
        print(f"Device VRAM: {vram_used / 1048576:.1f} / {vram_total / 1048576:.1f} MiB used")
    if gtt_used is not None and gtt_total is not None:
        print(f"Device GTT : {gtt_used / 1048576:.1f} / {gtt_total / 1048576:.1f} MiB used")
    print(f"{'PID':>8}  {'APP':<24} {'KFD VRAM':>10} {'DRM VRAM':>10} {'DRM GTT':>10} {'SHARED':>8}")
    for client in clients:
        kfd_vram = client.get("kfd_vram")
        kfd_mib = "-" if kfd_vram is None else f"{kfd_vram / 1048576:.1f}"
        vram_mib = client["vram"] / 1048576
        gtt_mib = client["gtt"] / 1048576
        if kfd_vram is None and vram_mib == 0 and gtt_mib == 0:
            continue
        name = client["name"][:24]
        print(f"{client['pid']:>8}  {name:<24} {kfd_mib:>10} {vram_mib:>10.1f} {gtt_mib:>10.1f} {len(client['shared_objects']):>8}")
    if kfd_warning:
        print(f"Warning: {kfd_warning}")
    print("KFD and DRM columns are separate accounting views; do not add them together.")
    print("Exported/shared GEM objects may appear under multiple processes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="AMD GPU PCI BDF (for example 0000:0a:00.0) or DRM card name")
    parser.add_argument("--input", help="parse a saved amdgpu_gem_info dump instead of reading debugfs")
    parser.add_argument("--kfd-input", help="parse saved amd-smi process JSON instead of querying amd-smi")
    args = parser.parse_args()

    try:
        if args.input:
            if args.device:
                bdfs = [resolve_device(args.device)]
            else:
                bdfs = ["fixture"]
            for bdf in bdfs:
                print_device(bdf, args.input, args.kfd_input)
            return 0

        if os.geteuid() != 0:
            print("error: debugfs process accounting requires root; run with sudo", file=sys.stderr)
            return 2
        bdfs = [resolve_device(args.device)] if args.device else discover_devices()
        if not bdfs:
            print("error: no AMD DRM devices found", file=sys.stderr)
            return 1
        for index, bdf in enumerate(bdfs):
            if index:
                print()
            print_device(bdf)
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
