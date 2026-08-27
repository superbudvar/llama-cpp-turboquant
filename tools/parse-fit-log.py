#!/usr/bin/env python3
"""Parse a llama.cpp (rpc-aware fit) server log and emit a per-run table: model,
fit params, context, layer spill, served n_ctx, and device memory at MODEL READY.

Handles both router-mode runs (delimited by "spawning server instance") and direct
single-model loads (delimited by the first model-load / fit line). Multiple runs in
one file are all reported.

Usage: tools/parse-fit-log.py [logfile]   (default: /tmp/test-kv2.log)
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test-kv2.log"

RE_RUN = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")
RE_MODEL = re.compile(r"load:\s+--model\s*$")
RE_CACHE = re.compile(r"load:\s+--cache-type-([kv])\s*$")
RE_CTX = re.compile(r"load:\s+--ctx-size\s*$")
RE_MINCTX = re.compile(r"load:\s+--fit-strategy-min-ctx\s*$")
RE_FIT = re.compile(r"load:\s+--fit-(strategy|target)\s*$")
RE_VALUE = re.compile(r"load:\s+(\S+)\s*$")
RE_SPILL = re.compile(r"spilling (\d+) layers to CPU to reach min-ctx (\d+) \(n_gpu_layers=(\d+) of (\d+)\)")
RE_NCTX = re.compile(r"llama_context:\s+n_ctx_seq\s+=\s+(\d+)")
RE_MODELREADY = re.compile(r"MODEL READY - device memory breakdown for '([^']+)':")
RE_DEV = re.compile(r"\|\s*-\s*(.+?)\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*\((.*?)\)")
RE_HOST = re.compile(r"\|\s*-\s*(.+?)\s*\|\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)")
RE_SYSFS = re.compile(r"ggml_rocm_sysfs_memory:\s*device (\d+) matched.*?vram_used=(\d+) MiB vram_total=(\d+) MiB vram_free=(\d+) MiB")
RE_TIMING = re.compile(r"print_timing:.*?\|\s*task \d+\s*\|\s*(\w[^,]*),\s*n_tokens\s*=\s*(\d+).*?/\s*([\d.]+) tokens per second")
RE_FITSTART = re.compile(r"fit_solve_rpc_aware:")

runs = []
cur = None
last_n_ctx = None
capture_breakdown_for = None
run_counter = 0


def new_run(name="run", port="?"):
    global cur, run_counter
    run_counter += 1
    cur = {
        "name": name, "port": port,
        "model": None, "cache_k": None, "cache_v": None, "ctx_size": None,
        "min_ctx": None, "fit_strategy": None, "fit_target": None,
        "spill": None, "n_gpu_layers": None, "n_layers": None,
        "served_n_ctx": None, "nctxs": [], "model_ready": {}, "sysfs": {}, "timing": [],
    }
    return cur


def flush():
    global cur
    if cur is None:
        return
    if cur["served_n_ctx"] is None and cur["nctxs"]:
        mx = max(cur["nctxs"])
        candidates = [v for v in cur["nctxs"] if v < mx]
        cur["served_n_ctx"] = max(candidates) if candidates else mx
    runs.append(cur)


for line in open(LOG, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")

    m = RE_RUN.search(line)
    if m:
        flush()
        new_run(m.group(1), m.group(2))
        continue

    # start a run for a direct (non-router) single-model load
    if cur is None and (RE_FITSTART.search(line) or RE_MODELREADY.search(line) or RE_MODEL.search(line)):
        new_run("direct-%d" % (run_counter + 1))
    if cur is None:
        continue

    mv = RE_VALUE.search(line)

    m_model = RE_MODEL.search(line)
    if m_model:
        cur["_pending"] = "model"
        continue
    m_cache = RE_CACHE.search(line)
    if m_cache:
        cur["_pending"] = "cache_" + m_cache.group(1)
        continue
    if RE_CTX.search(line):
        cur["_pending"] = "ctx_size"
        continue
    if RE_MINCTX.search(line):
        cur["_pending"] = "min_ctx"
        continue
    m_fit = RE_FIT.search(line)
    if m_fit:
        cur["_pending"] = "fit_" + m_fit.group(1)
        continue
    if "_pending" in cur:
        if mv is not None and not mv.group(1).startswith("--"):
            cur[cur.pop("_pending")] = mv.group(1)
        else:
            cur.pop("_pending", None)
        continue

    m = RE_SPILL.search(line)
    if m:
        cur["spill"] = int(m.group(1))
        cur["n_gpu_layers"] = int(m.group(3))
        cur["n_layers"] = int(m.group(4))
        continue

    m = RE_NCTX.search(line)
    if m:
        last_n_ctx = int(m.group(1))
        cur["nctxs"].append(last_n_ctx)
        continue

    m = RE_MODELREADY.search(line)
    if m:
        cur["served_n_ctx"] = last_n_ctx
        capture_breakdown_for = m.group(1)
        cur["model_ready"].setdefault(capture_breakdown_for, {})
        continue

    if capture_breakdown_for is not None:
        md = RE_DEV.search(line)
        if md:
            name = md.group(1).strip()
            _total, free = int(md.group(2)), int(md.group(3))
            inner = [int(x) for x in re.findall(r"\d+", md.group(4))]
            # inner = [self, model, ctx, comp]
            model, ctx, comp = (inner[1:] + [0, 0, 0])[:3]
            cur["model_ready"][capture_breakdown_for][name] = {
                "total": _total, "free": free, "model": model, "context": ctx, "compute": comp,
            }
            continue
        if RE_HOST.search(line):
            continue
        capture_breakdown_for = None

    m = RE_SYSFS.search(line)
    if m:
        dev, used, total, free = m.groups()
        cur["sysfs"][dev] = (int(used), int(total), int(free))
        continue

    m = RE_TIMING.search(line)
    if m:
        cur["timing"].append((m.group(1).strip(), int(m.group(2)), float(m.group(3))))
        continue

flush()

if not runs:
    print("no runs found in", LOG)
    sys.exit(0)

print("=" * 100)
for r in runs:
    print(f"RUN {r['name']}  (port {r['port']})")
    print(f"  model     : {r['model']}")
    print(f"  fit       : {r['fit_strategy']}  target={r['fit_target']}")
    print(f"  cache k/v : {r['cache_k']} / {r['cache_v']}")
    print(f"  ctx-size  : {r['ctx_size']}   min-ctx: {r['min_ctx']}")
    if r["spill"] is not None:
        print(f"  spill     : {r['spill']} layers to CPU  (n_gpu_layers={r['n_gpu_layers']} of {r['n_layers']})")
    print(f"  served ctx: {r['served_n_ctx']}")
    if r["model_ready"]:
        for mp, devs in r["model_ready"].items():
            print(f"  MODEL READY memory [{mp.split('/')[-1]}]:")
            for dev, d in devs.items():
                used = d["model"] + d["context"] + d["compute"]
                print(f"    {dev:28s} total={d['total']:6d}  used={used:6d} (model={d['model']} ctx={d['context']} comp={d['compute']})  free={d['free']:6d}")
    if r["sysfs"]:
        print(f"  sysfs (last seen per card):")
        for dev, (used, total, free) in sorted(r["sysfs"].items()):
            print(f"    card{dev}: used={used:6d}/{total:6d}  free={free:6d} MiB")
    if r["timing"]:
        for phase, ntok, tps in r["timing"]:
            print(f"  timing    : {phase:20s} n_tokens={ntok:6d}  {tps:.2f} tok/s")
    print("-" * 100)

print("=" * 100)
print(f"total runs parsed: {len(runs)}")
