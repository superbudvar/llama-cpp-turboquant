#!/usr/bin/env python3
"""Run a project build in the Libvirt Windows VM and export artifacts to the host."""

from __future__ import annotations

import argparse
import fcntl
import grp
import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


class BuildError(RuntimeError):
    pass


class GuestUnreachable(BuildError):
    pass


class HostKeyMismatch(BuildError):
    pass


def run(args: Sequence[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if check and result.returncode:
        raise BuildError(f"command failed ({result.returncode}): {args[0]}\n{result.stdout[-4000:]}")
    return result


def load_profile(path: Path) -> tuple[dict, Path]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise BuildError("unsupported project profile schema_version")
    for key in ("project", "domain", "virtiofs_tag", "guest_repo_root", "ssh", "scratch", "artifacts", "windows_build"):
        if key not in profile:
            raise BuildError(f"project profile is missing {key}")
    repo_root = (path.parent / profile.get("repo_root_from_profile", "../..")).resolve()
    if not (repo_root / ".git").exists():
        raise BuildError(f"source root is not a Git worktree: {repo_root}")
    return profile, repo_root


def parse_domifaddr(output: str, expected_mac: str | None = None) -> str | None:
    expected = expected_mac.lower() if expected_mac else None
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].lower() in {"name", "-"}:
            continue
        if expected and fields[1].lower() != expected:
            continue
        try:
            address = ipaddress.ip_interface(fields[-1]).ip
        except ValueError:
            continue
        if isinstance(address, ipaddress.IPv4Address):
            return str(address)
    return None


def parse_domblklist_targets(output: str) -> set[str]:
    targets: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] not in {"Type", "-"}:
            targets.add(fields[2])
    return targets


def parse_ed25519_keyscan(output: str, expected_fingerprint: str, host_key_alias: str) -> str:
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or fields[1] != "ssh-ed25519":
            continue
        key_line = host_key_alias + " " + " ".join(fields[1:3]) + "\n"
        fingerprint = subprocess.run(
            ["ssh-keygen", "-lf", "-"], input=" ".join(fields[1:3]) + "\n", text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
        )
        if fingerprint.returncode == 0 and expected_fingerprint in fingerprint.stdout:
            return key_line
    raise HostKeyMismatch("the VM ED25519 key did not match the pinned fingerprint")


def project_commit(repo_root: Path) -> str:
    commit = run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).stdout.strip()
    changed = run(["git", "-C", str(repo_root), "diff", "--name-only", "HEAD"]).stdout.splitlines()
    allowed = (".gitignore", "local-processes/")
    unexpected = [name for name in changed if not (name == ".gitignore" or name.startswith(allowed[1]))]
    if unexpected:
        raise BuildError("tracked source edits are present outside local-processes: " + ", ".join(unexpected[:12]))
    return commit


def existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def check_virtiofs_mapping(uri: str, domain: str, tag: str, repo_root: Path) -> None:
    xml_text = run(["virsh", "--connect", uri, "dumpxml", domain]).stdout
    root = ET.fromstring(xml_text)
    for fs in root.findall("./devices/filesystem"):
        target = fs.find("target")
        source = fs.find("source")
        if target is None or source is None or target.get("dir") != tag:
            continue
        source_dir = source.get("dir")
        if source_dir and Path(source_dir).resolve() == repo_root:
            return
        raise BuildError(f"virtiofs tag {tag} maps to {source_dir!r}, not {str(repo_root)!r}")
    raise BuildError(f"Libvirt domain {domain} has no virtiofs mapping for tag {tag}")


def host_ip_candidates(uri: str, domain: str, mac: str, sources: Sequence[str]) -> list[str]:
    addresses: list[str] = []
    for source in sources:
        result = run(["virsh", "--connect", uri, "domifaddr", domain, "--source", source], check=False, timeout=10)
        address = parse_domifaddr(result.stdout, mac)
        if address and address not in addresses:
            addresses.append(address)
    return addresses


def host_ip(uri: str, domain: str, mac: str, sources: Sequence[str], wait_seconds: int, poll_seconds: int) -> str | None:
    deadline = time.monotonic() + wait_seconds
    while True:
        addresses = host_ip_candidates(uri, domain, mac, sources)
        if addresses:
            return addresses[0]
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_seconds)


def cached_ip_path(scratch_root: Path, domain: str) -> Path:
    return scratch_root / "state" / f"{domain}.last-ip"


def read_cached_ip(path: Path) -> str | None:
    try:
        address = path.read_text(encoding="ascii").strip()
        return str(ipaddress.IPv4Address(address))
    except (OSError, ValueError):
        return None


def write_cached_ip(path: Path, address: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(address + "\n", encoding="ascii")
    tmp.replace(path)


def verify_artifacts(artifact_dir: Path, expected_executables: Sequence[str]) -> None:
    manifest_path = artifact_dir / "build-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"artifact manifest is missing or invalid: {manifest_path}") from exc
    if not isinstance(manifest.get("files"), list):
        raise BuildError(f"artifact manifest has no file records: {manifest_path}")
    recorded: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise BuildError("artifact manifest contains an invalid file record")
        name = item.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise BuildError("artifact manifest contains an invalid file name")
        path = artifact_dir / "bin" / name
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise BuildError(f"artifact missing or size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        expected_digest = item.get("sha256", "")
        if not isinstance(expected_digest, str) or digest.hexdigest().upper() != expected_digest.upper():
            raise BuildError(f"artifact SHA-256 mismatch: {path}")
        recorded.add(name.lower())
    missing = [name for name in expected_executables if name.lower() not in recorded]
    if missing:
        raise BuildError("artifact manifest is missing expected executables: " + ", ".join(missing))


def prepare_scratch_dir(root: Path, qemu_group: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    group = grp.getgrnam(qemu_group)
    if root.stat().st_uid != os.geteuid():
        raise BuildError(f"scratch root must be owned by the current user: {root}")
    os.chown(root, -1, group.gr_gid)
    root.chmod(0o2770)
    return group.gr_gid


def verify_host_key(ip: str, fingerprint: str, host_key_alias: str) -> str:
    scan = run(["ssh-keyscan", "-T", "5", "-t", "ed25519", ip], check=False, timeout=10)
    if scan.returncode and not scan.stdout.strip():
        raise GuestUnreachable(f"ssh-keyscan has no response from {ip}: {scan.stdout.strip()}")
    return parse_ed25519_keyscan(scan.stdout, fingerprint, host_key_alias)


def is_transient_ssh_failure(output: str) -> bool:
    lowered = output.lower()
    return any(message in lowered for message in (
        "connection refused", "connection timed out", "operation timed out", "no route to host",
        "connection reset", "kex_exchange_identification", "connection closed by remote host",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate configuration without starting or modifying the VM")
    args = parser.parse_args()

    scratch_file: Path | None = None
    known_hosts_file: Path | None = None
    attached = False
    started_by_runner = False
    remote_started = False
    remote_finished = False
    target = ""
    profile: dict = {}
    uri = "qemu:///system"
    domain = ""
    success = False
    keep_scratch = False
    try:
        profile, repo_root = load_profile(args.profile.resolve())
        uri = profile.get("libvirt_uri", "qemu:///system")
        domain = profile["domain"]
        commit = project_commit(repo_root)
        check_virtiofs_mapping(uri, domain, profile["virtiofs_tag"], repo_root)

        for executable in ("virsh", "qemu-img", "ssh", "ssh-keyscan", "ssh-keygen"):
            if shutil.which(executable) is None:
                raise BuildError(f"required host command not found: {executable}")

        scratch = profile["scratch"]
        scratch_size = int(scratch["size_gib"])
        reserve = int(scratch.get("reserve_bytes", 0))
        scratch_root = Path(scratch["host_root"]).expanduser()
        host_free = shutil.disk_usage(scratch_root.parent).free
        required_free = scratch_size * 1024**3 + reserve
        if host_free < required_free:
            raise BuildError(f"insufficient scratch space at {scratch_root.parent}: have {host_free} bytes, need at least {required_free}")
        qemu_gid = grp.getgrnam(scratch.get("qemu_group", "qemu")).gr_gid
        if qemu_gid not in os.getgroups() and os.getegid() != qemu_gid:
            raise BuildError(f"current user is not in the {scratch.get('qemu_group', 'qemu')} group needed for scratch access")
        if not os.access(scratch_root.parent, os.W_OK | os.X_OK):
            raise BuildError(f"scratch parent is not writable: {scratch_root.parent}")

        artifact_root = (repo_root / profile["artifacts"]["host_root_from_repo"]).resolve()
        artifact_free = shutil.disk_usage(existing_parent(artifact_root)).free
        minimum_artifact_free = int(profile["artifacts"].get("minimum_free_bytes", 0))
        if artifact_free < minimum_artifact_free:
            raise BuildError(f"insufficient host artifact space: have {artifact_free} bytes, need {minimum_artifact_free}")

        state = run(["virsh", "--connect", uri, "domstate", domain]).stdout.strip().lower()
        if state not in {"running", "shut off"}:
            raise BuildError(f"domain {domain} is in unsupported state: {state}")
        print(f"project={profile['project']} commit={commit} domain={domain} state={state}")
        print(f"source={repo_root} virtiofs_tag={profile['virtiofs_tag']} artifact_host={artifact_root}")
        print(f"scratch_host={scratch_root} virtual_size={scratch_size}GiB available={host_free} bytes")
        if args.dry_run:
            live_ips = host_ip_candidates(uri, domain, profile.get("interface_mac", ""), profile["ip_discovery"].get("sources", ["agent", "lease", "arp"])) if state == "running" else []
            cached_ip = read_cached_ip(cached_ip_path(scratch_root, domain))
            ip = live_ips[0] if live_ips else cached_ip
            if state == "running" and not ip:
                raise BuildError("dry-run cannot resolve the running VM IP from guest-agent, DHCP lease, ARP, or the last-known-IP cache")
            print(f"dry-run: would start if needed; current guest_ip={ip or 'resolved after boot'}; no scratch disk or guest changes made")
            return 0

        lock_root = Path.home() / ".cache" / "local-processes" / "locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{domain}.lock"
        with lock_path.open("w") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BuildError(f"another host-side build already holds {lock_path}") from exc

            state = run(["virsh", "--connect", uri, "domstate", domain]).stdout.strip().lower()
            if state == "shut off":
                run(["virsh", "--connect", uri, "start", domain])
                started_by_runner = True
                deadline = time.monotonic() + int(profile["ip_discovery"].get("wait_seconds", 180))
                while time.monotonic() < deadline:
                    state = run(["virsh", "--connect", uri, "domstate", domain]).stdout.strip().lower()
                    if state == "running":
                        break
                    time.sleep(2)
                if state != "running":
                    raise BuildError(f"Libvirt domain {domain} did not reach running state")

            ip_cfg = profile["ip_discovery"]
            ssh_cfg = profile["ssh"]
            key_path = Path(ssh_cfg["identity_file"]).expanduser()
            if not key_path.is_file():
                raise BuildError(f"SSH identity file not found: {key_path}")
            preflight_ps = [
                "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", profile["guest_build_script"], "-ProfilePath", profile["guest_profile_path"],
                "-ExpectedCommit", commit, "-PreflightOnly",
            ]
            preflight_cmd = subprocess.list2cmdline(preflight_ps)
            last_ip_file = cached_ip_path(scratch_root, domain)
            cached_ip = read_cached_ip(last_ip_file)
            mismatched_ips: set[str] = set()
            deadline = time.monotonic() + int(ip_cfg.get("wait_seconds", 180))
            ip = ""
            ssh_args: list[str] = []
            while not ip:
                candidates = host_ip_candidates(uri, domain, profile["interface_mac"], ip_cfg.get("sources", ["agent", "lease", "arp"]))
                if cached_ip and cached_ip not in candidates:
                    candidates.append(cached_ip)
                for candidate_ip in candidates:
                    if candidate_ip in mismatched_ips:
                        continue
                    try:
                        known_host_line = verify_host_key(candidate_ip, ssh_cfg["host_key_fingerprint"], ssh_cfg.get("host_key_alias", domain))
                    except GuestUnreachable:
                        continue
                    except HostKeyMismatch:
                        mismatched_ips.add(candidate_ip)
                        continue
                    with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False, dir="/tmp/opencode") as kh:
                        kh.write(known_host_line)
                        candidate_known_hosts = Path(kh.name)
                    candidate_ssh = [
                        "ssh", "-i", str(key_path), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={candidate_known_hosts}",
                        "-o", f"HostKeyAlias={ssh_cfg.get('host_key_alias', domain)}",
                        "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                        f"{ssh_cfg['user']}@{candidate_ip}",
                    ]
                    probe = run(candidate_ssh + ["hostname"], check=False, timeout=15)
                    if probe.returncode:
                        candidate_known_hosts.unlink(missing_ok=True)
                        if is_transient_ssh_failure(probe.stdout):
                            continue
                        raise BuildError(f"SSH to verified VM address {candidate_ip} failed:\n{probe.stdout[-2000:]}")
                    hostname_lines = [line.strip().lower() for line in probe.stdout.splitlines() if line.strip()]
                    expected_hostname = profile.get("guest_hostname", "").lower()
                    if expected_hostname and expected_hostname not in hostname_lines:
                        candidate_known_hosts.unlink(missing_ok=True)
                        mismatched_ips.add(candidate_ip)
                        continue
                    preflight = run(candidate_ssh + [preflight_cmd], check=False, timeout=60)
                    if preflight.returncode:
                        candidate_known_hosts.unlink(missing_ok=True)
                        if is_transient_ssh_failure(preflight.stdout):
                            continue
                        raise BuildError(f"Windows preflight failed at verified VM address {candidate_ip}:\n{preflight.stdout[-4000:]}")
                    ip = candidate_ip
                    ssh_args = candidate_ssh
                    known_hosts_file = candidate_known_hosts
                    print(preflight.stdout.strip())
                    qemu_gid = prepare_scratch_dir(scratch_root, scratch.get("qemu_group", "qemu"))
                    write_cached_ip(last_ip_file, ip)
                    break
                if ip:
                    break
                if time.monotonic() >= deadline:
                    raise BuildError(f"could not discover/authenticate an IPv4 address for {domain}; install/configure QEMU guest agent or add a DHCP reservation")
                time.sleep(int(ip_cfg.get("poll_seconds", 3)))

            qemu_gid = prepare_scratch_dir(scratch_root, scratch.get("qemu_group", "qemu"))
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            scratch_root = scratch_root / profile["project"]
            scratch_root.mkdir(parents=True, exist_ok=True)
            os.chown(scratch_root, -1, qemu_gid)
            scratch_root.chmod(0o2770)
            scratch_file = scratch_root / f"{profile['project']}-{commit[:12]}-{run_id}.qcow2"
            file_umask = os.umask(0o007)
            try:
                run(["qemu-img", "create", "-f", "qcow2", str(scratch_file), f"{scratch_size}G"])
            finally:
                os.umask(file_umask)
            os.chown(scratch_file, -1, qemu_gid)
            scratch_file.chmod(0o660)

            targets = parse_domblklist_targets(run(["virsh", "--connect", uri, "domblklist", domain, "--details"]).stdout)
            target = scratch.get("target", "sdb")
            if target in targets:
                scratch_file.unlink(missing_ok=True)
                raise BuildError(f"scratch target {target} is already attached to {domain}")
            serial = (scratch.get("serial_prefix", "WB") + uuid.uuid4().hex[:12]).upper()
            attach = ["virsh", "--connect", uri, "attach-disk", domain, str(scratch_file), target,
                      "--targetbus", scratch.get("target_bus", "scsi"), "--driver", "qemu", "--subdriver", "qcow2",
                      "--cache", "none", "--sourcetype", "file", "--serial", serial, "--live"]
            run(attach)
            attached = True

            ps = [
                "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", profile["guest_build_script"], "-ProfilePath", profile["guest_profile_path"],
                "-ExpectedCommit", commit, "-ScratchSerial", serial,
                "-ScratchDriveLetter", str(scratch.get("guest_drive_letter", "R")), "-RunId", run_id,
            ]
            remote_cmd = subprocess.list2cmdline(ps)
            print(f"building project={profile['project']} commit={commit} vm={domain} ip={ip} scratch={scratch_file}")
            remote_started = True
            result = run(ssh_args + [remote_cmd], check=False, timeout=int(profile.get("build_timeout_seconds", 14400)))
            remote_finished = True
            if result.stdout:
                print(result.stdout[-6000:])
            if result.returncode:
                keep_scratch = True
                raise BuildError(f"Windows build failed with exit code {result.returncode}; scratch retained at {scratch_file}")
            artifact_dir = artifact_root / commit[:12] / run_id
            try:
                verify_artifacts(artifact_dir, profile["artifacts"].get("expected_executables", []))
            except BuildError:
                keep_scratch = True
                raise
            success = True
            print(f"build completed; host artifacts: {artifact_dir}")

    except (BuildError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if known_hosts_file:
            known_hosts_file.unlink(missing_ok=True)
        if attached and domain and target:
            if remote_started and not remote_finished:
                print(f"ERROR: SSH ended before the remote build completed; leave scratch disk {target} attached and preserve {scratch_file}", file=sys.stderr)
                keep_scratch = True
                success = False
            else:
                detached = run(["virsh", "--connect", uri, "detach-disk", domain, target, "--live"], check=False, timeout=30)
                if detached.returncode:
                    print(f"ERROR: failed to detach scratch disk {target}; preserve scratch file {scratch_file}", file=sys.stderr)
                    keep_scratch = True
                    success = False
        if scratch_file and scratch_file.exists() and not keep_scratch:
            scratch_file.unlink(missing_ok=True)
        if started_by_runner and profile.get("shutdown_if_started", False):
            run(["virsh", "--connect", uri, "shutdown", domain], check=False, timeout=30)
            deadline = time.monotonic() + int(profile.get("shutdown_wait_seconds", 120))
            while time.monotonic() < deadline:
                state = run(["virsh", "--connect", uri, "domstate", domain], check=False, timeout=10).stdout.strip().lower()
                if state == "shut off":
                    break
                time.sleep(2)
            else:
                print(f"WARNING: Libvirt domain {domain} did not shut off before timeout", file=sys.stderr)

    return 0 if success or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
