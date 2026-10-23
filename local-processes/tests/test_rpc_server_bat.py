"""Dispatch tests for local-processes/windows/rpc-server.bat.

The launcher runs on Windows, so it cannot be exercised on the build host. It
can, however, be parsed and interpreted here: this file contains a small cmd
interpreter that understands the constructs the launcher uses (goto, exit /b,
shift, set, if, echo, for) and stops when the server is launched.

The bug this exists for: the first version had `:args_done` immediately
followed by `:usage`, so the label fell through into the usage text and the
script exited before doing anything - every run printed the usage and closed
the window. The tests below cover the dispatch and the fall-through class of
bug directly.
"""
import os
import re
import tempfile
import unittest
from pathlib import Path


BAT_PATH = Path(__file__).resolve().parents[1] / "windows" / "rpc-server.bat"
LABEL = re.compile(r"^:([A-Za-z_][A-Za-z0-9_]*)\s*$")


def logical_lines(text):
    """Join parenthesised blocks into single statements."""
    out, buf, depth = [], "", 0
    for raw in text.splitlines():
        stripped = raw.rstrip("\r").strip()
        if depth == 0 and not stripped:
            continue
        buf = stripped if depth == 0 else buf + " " + stripped
        # depth is the cumulative paren balance of the buffer, not a delta
        depth = len(re.findall(r"\(", buf)) - len(re.findall(r"\)", buf))
        if depth <= 0:
            out.append(buf)
            buf, depth = "", 0
    if buf:
        out.append(buf)
    return out


def split_amp(body):
    """Split on & that is not inside quotes."""
    parts, cur, quoted = [], [], False
    for ch in body:
        if ch == '"':
            quoted = not quoted
        if ch == "&" and not quoted:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


class BatchExit(Exception):
    def __init__(self, code, note=""):
        super().__init__(note)
        self.code = code
        self.note = note


class Batch:
    """A deliberately small cmd interpreter. Only what the launcher uses."""

    def __init__(self, path, cwd):
        with open(path, "r", newline="") as fh:
            self.lines = logical_lines(fh.read())
        self.cwd = cwd
        self.labels = {}
        for i, line in enumerate(self.lines):
            m = LABEL.match(line)
            if m:
                self.labels[m.group(1).lower()] = i
        self.vars = {}
        self.argv = []
        self.out = []
        self.launched = False

    def expand(self, text):
        def arg(m):
            idx = int(m.group(1)) - 1
            return self.argv[idx] if 0 <= idx < len(self.argv) else ""
        text = re.sub(r"%~(\d)", arg, text)

        def var(m):
            name = m.group(1)
            if name.upper() == "CD":
                return self.cwd
            return self.vars.get(name.upper(), m.group(0))
        return re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", var, text)

    def run_block(self, body):
        for part in split_amp(body):
            part = part.strip()
            if part:
                self.exec_stmt(part)

    def exec_stmt(self, stmt):
        stmt = stmt.strip()
        low = stmt.lower()
        if not stmt or low == "rem" or low.startswith("rem "):
            return
        if low in ("echo off", "@echo off"):
            return
        if low.startswith("setlocal") or low.startswith("endlocal"):
            return
        if low.startswith("exit /b"):
            code = self.expand(stmt.strip()[len("exit /b"):]).strip()
            raise BatchExit(int(code) if code.isdigit() else 0)
        if low.startswith("goto "):
            self.pc = self.label(stmt.strip()[5:].strip())
            return
        if low.startswith("shift"):
            if self.argv:
                self.argv.pop(0)
            return
        if low.startswith("set "):
            m = re.match(r'^set\s+"([^=]+)=(.*)"$', stmt, re.S)
            if m:
                name, val = m.group(1).strip().upper(), self.expand(m.group(2))
            else:
                m = re.match(r"^set\s+([^=]+)=(.*)$", stmt, re.S)
                if not m:
                    return
                name, val = m.group(1).strip().upper(), self.expand(m.group(2))
            if name:
                self.vars[name] = val
            else:
                self.out.append(val)
            return
        if low.startswith("for "):
            m = re.search(r'do\s+set\s+"?([A-Za-z_][A-Za-z0-9_]*)=', stmt, re.I)
            if m and m.group(1).upper() == "RPC_DIR":
                self.vars["RPC_DIR"] = self.cwd
            return
        if low.startswith("if "):
            self.exec_if(stmt[3:].strip())
            return
        if low.startswith("echo "):
            body = re.sub(r"^[>\s]+", "", stmt[5:])
            self.out.append(self.expand(body))
            return
        if low.startswith("powershell"):
            self.launched = True
            raise BatchExit(-1, "server launch reached")

    def label(self, name):
        key = name.lower()
        if key not in self.labels:
            raise BatchExit(99, "goto to unknown label :" + name)
        return self.labels[key]

    def exec_if(self, body):
        negate = False
        if body.lower().startswith("not "):
            negate = True
            body = body[4:].strip()
        if body.lower().startswith("/i "):
            body = body[3:].strip()
        if body.lower().startswith("exist "):
            path = self.expand(body[6:].strip().strip('"'))
            ok = os.path.exists(path)
            self.run_when(body, ok != negate)
            return
        m = re.match(r'^("(?:[^"]*)"|\S+)\s*==\s*("(?:[^"]*)"|\S+)\s*(.*)$', body)
        if not m:
            raise BatchExit(98, "cannot evaluate if: " + body)
        left = self.expand(m.group(1)).strip('"')
        right = self.expand(m.group(2)).strip('"')
        ok = left == right
        if negate:
            ok = not ok
        self.run_when(m.group(3), ok)

    def run_when(self, rest, ok):
        rest = rest.strip()
        if not rest:
            return
        if rest.lower().startswith("goto "):
            if ok:
                self.pc = self.label(rest[5:].strip())
            return
        if rest.startswith("("):
            inner = rest[1:]
            if inner.endswith(")"):
                inner = inner[:-1]
            if ok:
                self.run_block(inner)
            return
        if ok:
            self.exec_stmt(rest)

    def run(self, argv):
        self.argv = list(argv)
        self.pc = 0
        try:
            while 0 <= self.pc < len(self.lines):
                stmt = self.lines[self.pc]
                self.pc += 1
                self.exec_stmt(stmt)
        except BatchExit as exc:
            return exc
        return BatchExit(0)


class RpcServerBatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # the launcher looks for the exe next to itself, so fake one
        Path(cls.tmp.name, "ggml-rpc-server.exe").write_bytes(b"")
        cls.bat = Batch(BAT_PATH, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_bat(self, argv):
        bat = Batch(BAT_PATH, self.tmp.name)
        exit_info = bat.run(argv)
        return exit_info, bat

    def assert_launched(self, argv):
        exit_info, bat = self.run_bat(argv)
        self.assertTrue(bat.launched,
                        "the launcher never reached the server for args %r" % (argv,))
        self.assertFalse(any(o.startswith("Usage:") for o in bat.out),
                         "the launcher printed the usage text for args %r" % (argv,))
        return bat

    # ------------------------------------------------------------- dispatch
    def test_no_arguments_starts_the_server(self):
        bat = self.assert_launched([])
        self.assertEqual(bat.vars.get("RPC_PORT"), "50052")
        self.assertEqual(bat.vars.get("RPC_HOST"), "0.0.0.0")

    def test_default_keeps_tq4_1s_native(self):
        bat = self.assert_launched([])
        self.assertEqual(bat.vars.get("GGML_TQ_NATIVE"), "1")

    def test_tq_native_can_be_turned_off(self):
        bat = self.assert_launched(["--tq-native=0"])
        self.assertEqual(bat.vars.get("GGML_TQ_NATIVE"), "0")

    def test_tq_native_can_be_forced_on(self):
        bat = self.assert_launched(["--tq-native=1"])
        self.assertEqual(bat.vars.get("GGML_TQ_NATIVE"), "1")

    def test_port_option(self):
        bat = self.assert_launched(["-p", "1234"])
        self.assertEqual(bat.vars.get("RPC_PORT"), "1234")
        bat = self.assert_launched(["--port", "1234"])
        self.assertEqual(bat.vars.get("RPC_PORT"), "1234")

    def test_host_option(self):
        bat = self.assert_launched(["-H", "1.2.3.4"])
        self.assertEqual(bat.vars.get("RPC_HOST"), "1.2.3.4")

    def test_threads_option(self):
        bat = self.assert_launched(["-t", "8"])
        self.assertEqual(bat.vars.get("RPC_THREADS"), "8")

    def test_device_option(self):
        bat = self.assert_launched(["-d", "CUDA0"])
        self.assertEqual(bat.vars.get("RPC_DEVICE"), "CUDA0")

    def test_foreground_option(self):
        bat = self.assert_launched(["--foreground"])
        self.assertEqual(bat.vars.get("RPC_FOREGROUND"), "1")

    def test_options_combine(self):
        bat = self.assert_launched(["-p", "1234", "-t", "8", "--tq-native=0"])
        self.assertEqual(bat.vars.get("RPC_PORT"), "1234")
        self.assertEqual(bat.vars.get("RPC_THREADS"), "8")
        self.assertEqual(bat.vars.get("GGML_TQ_NATIVE"), "0")

    # --------------------------------------------------------------- errors
    def test_help_prints_usage_and_exits_zero(self):
        exit_info, bat = self.run_bat(["--help"])
        self.assertEqual(exit_info.code, 0)
        self.assertFalse(bat.launched)
        self.assertTrue(any(o.startswith("Usage:") for o in bat.out))

    def test_unknown_option_is_rejected(self):
        exit_info, bat = self.run_bat(["--nope"])
        self.assertEqual(exit_info.code, 2)
        self.assertFalse(bat.launched)

    def test_option_without_a_value_is_rejected(self):
        exit_info, bat = self.run_bat(["-p"])
        self.assertEqual(exit_info.code, 2)
        self.assertFalse(bat.launched)

    # ------------------------------------------------------- static checks
    def test_every_goto_target_exists(self):
        text = BAT_PATH.read_text(newline="")
        labels = {m.group(1).lower() for m in (LABEL.match(l) for l in text.splitlines()) if m}
        targets = set(re.findall(r"goto\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.I))
        missing = {t for t in targets if t.lower() not in labels}
        self.assertEqual(missing, set(), "goto targets with no label: %s" % sorted(missing))

    def test_no_label_falls_through_into_the_usage_text(self):
        """The bug that shipped first: :args_done sat right above :usage."""
        text = BAT_PATH.read_text(newline="")
        lines = logical_lines(text)
        label_at = {}
        for i, line in enumerate(lines):
            m = LABEL.match(line)
            if m:
                label_at[m.group(1).lower()] = i

        def first_statement(start):
            for line in lines[start + 1:]:
                if line.lower().startswith("rem "):
                    continue
                return line.lower()
            return ""

        offenders = []
        for name, start in label_at.items():
            end = min([s for s in label_at.values() if s > start] or [len(lines)])
            block = [l for l in lines[start + 1:end] if not l.lower().startswith("rem ")]
            last = block[-1].lower() if block else ""
            if last.startswith("goto ") or last.startswith("exit /b"):
                continue
            # the block ends without terminating, so control falls into the
            # next label. That is only a bug when the next label is the usage
            # text, which must be reachable by an explicit goto only.
            if end < len(lines) and first_statement(end).startswith("echo usage:"):
                offenders.append(name)
        self.assertEqual(offenders, [],
                         "these labels fall through into the usage text: %s" % offenders)

    def test_file_uses_crlf_line_endings(self):
        data = BAT_PATH.read_bytes()
        self.assertEqual(data.count(b"\n") - data.count(b"\r\n"), 0,
                         "cmd.exe needs CRLF line endings for labels and goto")


if __name__ == "__main__":
    unittest.main()
