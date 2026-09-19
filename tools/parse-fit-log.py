#!/usr/bin/env python3
"""Parse a llama.cpp (rpc-aware fit) server log and emit a per-run table: model,
fit params, context, layer spill, served n_ctx, offload ratio, KV/RS per device,
device memory at MODEL READY, eval timing, spec/MoE status.

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
RE_OFFLOAD_NEW = re.compile(r"load_tensors: offloading (\d+) layers to GPU \((\d+) transformer \+ (\d+) output\)")
RE_OFFLOAD_OLD = re.compile(r"load_tensors: offloading (\d+) layers to GPU(?!.*\()")
RE_OFFLOAD_SIMPLE = re.compile(r"load_tensors: (not offloading any layers|offloading output layer) to GPU")
RE_OFFLOAD_DONE = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
RE_TRAINCTX = re.compile(r"print_info: n_ctx_train\s*=\s*(\d+)")
RE_KV = re.compile(r"llama_kv_cache:\s*(.+?) KV buffer size\s*=\s*([\d.]+) MiB")
RE_RS = re.compile(r"llama_memory_recurrent:\s*(.+?) RS buffer size\s*=\s*([\d.]+) MiB")
RE_MOE = re.compile(r"sched_reserve: MoE cache requested=(\w+) resolved=(\w+)")
RE_SPEC = re.compile(r"common_speculative_init_result: (loading draft model|creating MTP draft context) '?([^'\n]+)")
RE_EVAL_TIMING = re.compile(r"slot print_timing:.*?task \d+\s*\|\s*(prompt eval|eval) time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+) tokens\s*\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second\)")

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
        "offload": None, "offload_transformer": None, "offload_output": None,
        "spill": None, "n_gpu_layers": None, "n_layers": None,
        "served_n_ctx": None, "nctxs": [], "n_ctx_train": None,
        "kv": {}, "rs": {}, "moe_requested": None, "moe_resolved": None,
        "spec": None, "model_ready": {}, "sysfs": {}, "timing": [],
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

    m = RE_OFFLOAD_NEW.search(line)
    if m:
        # m.group(1)=offloading N, group(2)=transformer, group(3)=output
        # the RE_OFFLOAD_DONE line sets cur["offload"]; here we just capture breakdown
        cur["offload_transformer"] = int(m.group(2))
        cur["offload_output"] = int(m.group(3))
        continue

    m = RE_OFFLOAD_OLD.search(line)
    if m:
        cur["offload"] = "%s/%s" % (m.group(1), m.group(2))
        continue

    m = RE_OFFLOAD_SIMPLE.search(line)
    if m:
        cur["offload"] = "0/%s" % (cur.get("n_layers", "?"))
        continue

    m = RE_OFFLOAD_DONE.search(line)
    if m:
        cur["offload"] = "%s/%s" % (m.group(1), m.group(2))
        continue

    m = RE_TRAINCTX.search(line)
    if m:
        cur["n_ctx_train"] = int(m.group(1))
        continue

    m = RE_KV.search(line)
    if m:
        cur["kv"][m.group(1).strip()] = float(m.group(2))
        continue

    m = RE_RS.search(line)
    if m:
        cur["rs"][m.group(1).strip()] = float(m.group(2))
        continue

    m = RE_MOE.search(line)
    if m:
        cur["moe_requested"] = m.group(1)
        cur["moe_resolved"] = m.group(2)
        continue

    m = RE_SPEC.search(line)
    if m:
        cur["spec"] = m.group(2).split("/")[-1]
        continue

    m = RE_EVAL_TIMING.search(line)
    if m:
        phase, ms, ntok, tps = m.groups()
        cur["timing"].append(("%s eval" % phase, int(ntok), float(tps)))
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
        mh = RE_HOST.search(line)
        if mh:
            name = mh.group(1).strip()
            # host layout: total = free + model + compute  (no context column)
            _total, free, model, comp = (int(mh.group(i)) for i in range(2, 6))
            cur["model_ready"][capture_breakdown_for][name] = {
                "total": _total, "free": free, "model": model, "context": 0, "compute": comp,
            }
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
    offload_str = r['offload'] if r['offload'] is not None else "?"
    if r.get("offload_transformer") is not None and r.get("offload_output") is not None:
        offload_str += f" ({r['offload_transformer']} trans + {r['offload_output']} out)"
    print(f"  offload   : {offload_str} layers to GPU")
    print(f"  train ctx : {r['n_ctx_train']}")
    print(f"  served ctx: {r['served_n_ctx']}")
    if r["kv"] or r["rs"]:
        print(f"  KV buffer : {', '.join(f'{d}={v:.0f}' for d, v in sorted(r['kv'].items()))}")
    if r["rs"]:
        print(f"  RS buffer : {', '.join(f'{d}={v:.0f}' for d, v in sorted(r['rs'].items()))}")
    if r["moe_requested"]:
        print(f"  MoE cache : requested={r['moe_requested']} resolved={r['moe_resolved']}")
    if r["spec"]:
        print(f"  spec draft: {r['spec']}")
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
        seen = {}
        for phase, ntok, tps in r["timing"]:
            seen.setdefault(phase, []).append((ntok, tps))
        for phase in sorted(seen):
            entries = seen[phase]
            if phase == "prompt processing":
                keep = [e for e in entries if e[0] > 0]
                if not keep:
                    continue
                ntok, tps = max(keep, key=lambda e: e[0])
                print(f"  timing    : {phase:20s} n_tokens={ntok:6d}  {tps:.2f} tok/s  (best of {len(entries)})")
            else:
                print(f"  timing    : {phase:20s} n_tokens={entries[-1][0]:6d}  {entries[-1][1]:.2f} tok/s")
    print("-" * 100)

print("=" * 100)
print(f"total runs parsed: {len(runs)}")
