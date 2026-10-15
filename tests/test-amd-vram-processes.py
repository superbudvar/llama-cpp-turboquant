#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "amd-vram-processes.py"
SPEC = importlib.util.spec_from_file_location("amd_vram_processes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParseGemInfoTests(unittest.TestCase):
    def test_groups_vram_and_gtt_bytes_by_process(self):
        text = """pid 123 command app-one:
        0x00000001: 1048576 byte VRAM NO_CPU_ACCESS
        0x00000002: 2097152 byte GTT CPU_ACCESS_REQUIRED
        0x00000003: 4096 byte CPU CPU_ACCESS_REQUIRED
pid 456 command app-two:
        0x00000001: 524288 byte VRAM exported as ino:77 NO_CPU_ACCESS
"""
        clients = MODULE.parse_gem_info(text)
        self.assertEqual((clients[123]["name"], clients[123]["vram"], clients[123]["gtt"]),
                         ("app-one", 1048576, 2097152))
        self.assertEqual((clients[456]["name"], clients[456]["vram"], clients[456]["gtt"]),
                         ("app-two", 524288, 0))

    def test_deduplicates_repeated_export_within_process(self):
        text = """pid 123 command app-one:
        0x00000001: 1048576 byte VRAM exported as ino:77
        0x00000002: 1048576 byte VRAM exported as ino:77
"""
        clients = MODULE.parse_gem_info(text)
        self.assertEqual(clients[123]["vram"], 1048576)
        self.assertEqual(len(clients[123]["shared_objects"]), 1)

    def test_parses_amd_smi_process_usage(self):
        text = """[
          {"gpu": 0, "process_list": [
            {"process_info": {"name": "/usr/bin/llama-server", "pid": 824183,
              "mem_usage": {"value": 14806278144, "unit": "B"}}},
            {"process_info": {"name": "/usr/bin/Xorg", "pid": 3243,
              "mem_usage": {"value": 0, "unit": "B"}}}
          ]}
        ]"""
        clients = MODULE.parse_kfd_processes(text)
        self.assertEqual(clients[824183], {
            "pid": 824183,
            "name": "llama-server",
            "kfd_vram": 14806278144,
        })
        self.assertEqual(clients[3243]["kfd_vram"], 0)


if __name__ == "__main__":
    unittest.main()
