#!/usr/bin/env python3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PARSER = Path(__file__).resolve().parents[1] / "tools" / "parse-fit-log.py"


class ParseFitLogTests(unittest.TestCase):
    def test_reports_mtp_refits_and_current_trace_probes(self):
        log = """\
0.00.001.000 I srv load: spawning server instance with name=Qwen3.8-27B-Q3 on port 12345
0.00.002.000 I srv load:   --model
0.00.003.000 I srv load:   /models/qwen.gguf
0.00.004.000 I srv load:   --fit-strategy
0.00.005.000 I srv load:   rpc-aware
0.00.006.000 I srv load:   --fit-strategy-min-ctx
0.00.007.000 I srv load:   160000
[12345] [FITDBG] rpc-aware entered nd=2 local=1 loader_ngl=65 min_ctx=160000
[12345] [FITDBG] building candidates: 2 dense_splits, 2 pool_opts, moe.n_expert=256
[12345] 0.10.001.000 I fit_solve_rpc_aware: rpc_frac=0.3353 min_ctx measure=1 devs=1 margin=1 -> FEASIBLE (fraction=1/12 probes=1/32 candidate=dense=ROCm0,pool=RPC0)
[12345] 0.10.002.000 I fit_grow_context: grow fits n_ctx=262144 measure=1 margins=1 -> ok (probes=2/32)
[12345] 0.10.003.000 I fit_solve_rpc_aware: selected candidate=dense=ROCm0,pool=RPC0 n_ctx=262144 rpc_frac=0.3353 split_size=2 (min=160000)
[12345] 0.10.004.000 I common_fit_strategy_apply_impl: successfully fit params to free device memory
[12345] 0.10.005.000 I load_model: [spec] target-context preflight iter=1 n_ctx=164608 MTP=1049.66 MiB reserve_grew=1
[12345] [FITDBG] rpc-aware entered nd=2 local=1 loader_ngl=65 min_ctx=160000
[12345] [FITDBG] building candidates: 2 dense_splits, 2 pool_opts, moe.n_expert=256
[12345] 0.20.001.000 I fit_solve_rpc_aware: rpc_frac=0.3125 min_ctx measure=1 devs=1 margin=1 -> FEASIBLE (fraction=2/12 probes=4/32 candidate=dense=ROCm0,pool=RPC0)
[12345] 0.20.002.000 I fit_grow_context: grow fits n_ctx=180224 measure=1 margins=0 -> FAIL (probes=5/32)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit.log"
            path.write_text(log, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PARSER), str(path)],
                check=True, capture_output=True, text=True,
            )

        self.assertIn("rpc-aware passes: 2", result.stdout)
        self.assertIn("#1: successfully fit params", result.stdout)
        self.assertIn("selected: dense=ROCm0,pool=RPC0 n_ctx=262144 rpc_frac=0.3353", result.stdout)
        self.assertIn("MTP=1049.66 MiB (reserve grew)", result.stdout)
        self.assertIn("#2: in progress", result.stdout)
        self.assertIn("logged split/context probes=1/1", result.stdout)
        self.assertIn("latest context probe: n_ctx=180224 FAIL probes=5/32", result.stdout)


if __name__ == "__main__":
    unittest.main()
