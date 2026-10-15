import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_vm.py"
SPEC = importlib.util.spec_from_file_location("build_vm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
build_vm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_vm)


class VirshOutputTests(unittest.TestCase):
    def test_domifaddr_selects_expected_mac_and_ipv4(self):
        output = """
 Name       MAC address          Protocol     Address
-------------------------------------------------------------------------------
 vnet0      52:54:00:13:c3:a7    ipv4         192.168.4.194/0
 vnet1      52:54:00:13:c3:a8    ipv6         fe80::1/64
"""
        self.assertEqual(build_vm.parse_domifaddr(output, "52:54:00:13:c3:a7"), "192.168.4.194")
        self.assertIsNone(build_vm.parse_domifaddr(output, "52:54:00:13:c3:ff"))

    def test_domblklist_targets(self):
        output = """
 Type    Device   Target   Source
 block   disk     sda      /dev/seclin/windows11
 file    cdrom    sdc      /var/lib/libvirt/images/virtio-win.iso
"""
        self.assertEqual(build_vm.parse_domblklist_targets(output), {"sda", "sdc"})

    def test_cached_ip_is_normalized_and_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state" / "win11.last-ip"
            build_vm.write_cached_ip(path, "192.168.4.194")
            self.assertEqual(build_vm.read_cached_ip(path), "192.168.4.194")
            path.write_text("not-an-ip\n", encoding="ascii")
            self.assertIsNone(build_vm.read_cached_ip(path))

    def test_ssh_boot_races_are_retryable_but_auth_errors_are_not(self):
        self.assertTrue(build_vm.is_transient_ssh_failure("ssh: connect to host 192.0.2.1 port 22: Connection refused"))
        self.assertTrue(build_vm.is_transient_ssh_failure("Connection timed out"))
        self.assertFalse(build_vm.is_transient_ssh_failure("Permission denied (publickey)"))

    def test_artifact_verification_checks_required_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            bin_dir = artifact_dir / "bin"
            bin_dir.mkdir()
            binary = bin_dir / "llama-server.exe"
            binary.write_bytes(b"validated executable")
            manifest = {
                "files": [{
                    "name": binary.name,
                    "bytes": binary.stat().st_size,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest().upper(),
                }]
            }
            (artifact_dir / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            build_vm.verify_artifacts(artifact_dir, ["llama-server.exe"])
            binary.write_bytes(b"tampered")
            with self.assertRaises(build_vm.BuildError):
                build_vm.verify_artifacts(artifact_dir, ["llama-server.exe"])

    @patch.object(build_vm.subprocess, "run")
    def test_host_key_pin_rewrites_ip_to_stable_alias(self, run_mock):
        run_mock.return_value = type("Result", (), {
            "returncode": 0,
            "stdout": "256 SHA256:wVzFFS03nA7J03osC/QjoLNvESCjpt16flS6agODNoE testkey (ED25519)\n",
        })()
        keyscan = "192.168.4.194 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey\n"
        result = build_vm.parse_ed25519_keyscan(
            keyscan,
            "SHA256:wVzFFS03nA7J03osC/QjoLNvESCjpt16flS6agODNoE",
            "win11-build",
        )
        self.assertEqual(result, "win11-build ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey\n")

    @patch.object(build_vm.subprocess, "run")
    def test_host_key_mismatch_fails_closed(self, run_mock):
        run_mock.return_value = type("Result", (), {"returncode": 0, "stdout": "256 SHA256:wrong key\n"})()
        keyscan = "192.168.4.194 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey\n"
        with self.assertRaises(build_vm.BuildError):
            build_vm.parse_ed25519_keyscan(keyscan, "SHA256:expected", "win11-build")


if __name__ == "__main__":
    unittest.main()
