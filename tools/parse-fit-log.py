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
RE_MTP_PREFLIGHT = re.compile(r"target-context preflight iter=(\d+) n_ctx=(\d+) MTP=([\d.]+) MiB reserve_grew=(\d+)")
RE_UPSTREAM_FIT_OK = re.compile(r"common_fit_params: successfully fit params to free device memory")
RE_UPSTREAM_FIT_TIME = re.compile(r"common_fit_params: fitting params to free memory took ([\d.]+) seconds")
RE_RPC_FIT_START = re.compile(r"\[FITDBG\] rpc-aware entered nd=(\d+) local=(\d+) loader_ngl=(\d+) min_ctx=(\d+)")
RE_RPC_CANDIDATES = re.compile(r"\[FITDBG\] building candidates: (\d+) dense_splits, (\d+) pool_opts, moe\.n_expert=(\d+)")
RE_RPC_FRAC = re.compile(r"fit_solve_rpc_aware: rpc_frac=([\d.]+)(?: min_ctx)? measure=(\d+) devs=(\d+) margin=(\d+) -> (FEASIBLE|rejected)")
RE_GROW_RESULT = re.compile(r"grow fits n_ctx=(\d+) measure=(\d+) margins=(\d+) -> (ok|FAIL)(?: \(probes=(\d+)/(\d+)\))?")
RE_CANDIDATE_REJECT = re.compile(r"fit_solve_rpc_aware: candidate (.+?) cannot fit even the minimum context, skipping")
RE_CANDIDATE_SELECTED = re.compile(r"fit_solve_rpc_aware: selected candidate=(.+?) n_ctx=(\d+) rpc_frac=([\d.]+) split_size=(\d+)")
RE_MARGIN = re.compile(r"common_fit_strategy_apply_impl: device\[(\d+)\]=(\S+) fit-target=(-?\d+) MiB base-margin=(-?\d+) MiB")
RE_STRATEGY_RESULT = re.compile(r"common_fit_strategy_apply_impl: (successfully fit params|failed to fit params|encountered an error).*")
RE_EXIT = re.compile(r"instance name=(\S+) exited with status (\d+)")
RE_LOAD_PROGRESS = re.compile(r'cmd_child_to_router:state:\{"state":"(\w+)","payload":\{"stages":\[.*?\],"current":"(\w+)","value":([\d.]+)\}\}')
RE_CACHE_EVENT = re.compile(r"fit cache (?:miss|hit)|selected fit cache save|fit cache save|saved cache:|fit_delta_refresh: (?:no cache|found|refreshed)|discrepancy feedback cache save|updating EMA|skipped: projected_free", re.IGNORECASE)

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
        "upstream_fit_attempts": [], "rpc_fit": None, "load_status": "incomplete",
        "rpc_fits": [], "mtp_preflights": [], "load_progress": None,
        "cache_events": [], "fit_margins": [],
    }
    return cur


def flush():
    global cur
    if cur is None:
        return
    runs.append(cur)


for line in open(LOG, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")

    m = RE_RUN.search(line)
    if m:
        flush()
        new_run(m.group(1), m.group(2))
        last_n_ctx = None
        capture_breakdown_for = None
        continue

    # start a run for a direct (non-router) single-model load
    if cur is None and (RE_FITSTART.search(line) or RE_MODELREADY.search(line) or RE_MODEL.search(line)):
        new_run("direct-%d" % (run_counter + 1))
    if cur is None:
        continue

    if RE_CACHE_EVENT.search(line):
        cur["cache_events"].append(line.strip())

    m = RE_MARGIN.search(line)
    if m:
        device, name, target, margin = m.groups()
        cur["fit_margins"].append((int(device), name, int(target), int(margin)))
        continue

    m = RE_LOAD_PROGRESS.search(line)
    if m:
        cur["load_progress"] = (m.group(1), m.group(2), float(m.group(3)))
        cur["load_status"] = m.group(1)
        continue

    m = RE_MTP_PREFLIGHT.search(line)
    if m:
        iteration, n_ctx, mtp_mib, reserve_grew = m.groups()
        cur["mtp_preflights"].append({
            "iteration": int(iteration), "n_ctx": int(n_ctx),
            "mtp_mib": float(mtp_mib), "reserve_grew": bool(int(reserve_grew)),
        })
        continue

    m = RE_UPSTREAM_FIT_OK.search(line)
    if m:
        cur["upstream_fit_attempts"].append({"result": "success", "seconds": None})
        continue
    m = RE_UPSTREAM_FIT_TIME.search(line)
    if m:
        seconds = float(m.group(1))
        if cur["upstream_fit_attempts"] and cur["upstream_fit_attempts"][-1]["seconds"] is None:
            cur["upstream_fit_attempts"][-1]["seconds"] = seconds
        else:
            cur["upstream_fit_attempts"].append({"result": "no success marker", "seconds": seconds})
        continue
    m = RE_RPC_FIT_START.search(line)
    if m:
        if cur["rpc_fit"] is not None and cur["rpc_fit"]["result"] == "in progress":
            cur["rpc_fit"]["result"] = "superseded by later fit pass"
        cur["rpc_fit"] = {
            "min_ctx": int(m.group(4)), "dense_splits": None, "pool_opts": None,
            "n_expert": None, "rpc_fracs": [], "grow": [],
            "rejected_candidates": [], "selected": None, "result": "in progress",
        }
        cur["rpc_fits"].append(cur["rpc_fit"])
        continue
    if cur["rpc_fit"] is not None:
        m = RE_RPC_CANDIDATES.search(line)
        if m:
            dense_splits, pool_opts, n_expert = m.groups()
            cur["rpc_fit"]["dense_splits"] = int(dense_splits)
            cur["rpc_fit"]["pool_opts"] = int(pool_opts)
            cur["rpc_fit"]["n_expert"] = int(n_expert)
            continue
        m = RE_RPC_FRAC.search(line)
        if m:
            frac, measured, devices, margin, verdict = m.groups()
            cur["rpc_fit"]["rpc_fracs"].append({
                "frac": float(frac), "measured": int(measured), "devices": int(devices),
                "margin_ok": int(margin), "verdict": verdict,
            })
            continue
        m = RE_GROW_RESULT.search(line)
        if m:
            n_ctx, measured, margins_ok, verdict, probes_used, probes_limit = m.groups()
            cur["rpc_fit"]["grow"].append({
                "n_ctx": int(n_ctx), "measured": int(measured),
                "margins_ok": int(margins_ok), "verdict": verdict,
                "probes_used": int(probes_used) if probes_used is not None else None,
                "probes_limit": int(probes_limit) if probes_limit is not None else None,
            })
            continue
        m = RE_CANDIDATE_REJECT.search(line)
        if m:
            cur["rpc_fit"]["rejected_candidates"].append(m.group(1))
            continue
        m = RE_CANDIDATE_SELECTED.search(line)
        if m:
            candidate, n_ctx, rpc_frac, split_size = m.groups()
            cur["rpc_fit"]["selected"] = {
                "candidate": candidate, "n_ctx": int(n_ctx),
                "rpc_frac": float(rpc_frac), "split_size": int(split_size),
            }
            cur["rpc_fit"]["result"] = "selected"
            continue
        m = RE_STRATEGY_RESULT.search(line)
        if m:
            cur["rpc_fit"]["result"] = m.group(1)
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
        cur["load_status"] = "MODEL READY"
        capture_breakdown_for = m.group(1)
        cur["model_ready"].setdefault(capture_breakdown_for, {})
        continue

    m = RE_EXIT.search(line)
    if m and m.group(1) == cur["name"]:
        cur["load_status"] = "exited with status %s" % m.group(2)
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
    max_probed_ctx = max(r["nctxs"]) if r["nctxs"] else None
    print(f"  load state: {r['load_status']}")
    print(f"  served ctx: {r['served_n_ctx'] if r['served_n_ctx'] is not None else 'unconfirmed'}")
    print(f"  max probed: {max_probed_ctx}")
    over_train_ctx = sorted(set(n for n in r["nctxs"] if r["n_ctx_train"] and n > r["n_ctx_train"]))
    if over_train_ctx:
        print(f"  over-train probes: {', '.join(map(str, over_train_ctx))}")
    if r["load_progress"]:
        state, stage, value = r["load_progress"]
        print(f"  progress  : {state} / {stage} {value:.1%}")
    for i, attempt in enumerate(r["upstream_fit_attempts"], 1):
        elapsed = "?" if attempt["seconds"] is None else "%.2fs" % attempt["seconds"]
        print(f"  upstream fit #{i}: {attempt['result']} ({elapsed})")
    if r["mtp_preflights"]:
        print("  MTP preflight:")
        for item in r["mtp_preflights"]:
            reserve = "reserve grew" if item["reserve_grew"] else "reserve stable"
            print(f"    #{item['iteration']}: n_ctx={item['n_ctx']} MTP={item['mtp_mib']:.2f} MiB ({reserve})")
    if r["rpc_fits"]:
        print(f"  rpc-aware passes: {len(r['rpc_fits'])}")
        for fit_index, fit in enumerate(r["rpc_fits"], 1):
            counts = "candidate counts pending"
            if fit["dense_splits"] is not None:
                counts = f"{fit['dense_splits']} dense splits, {fit['pool_opts']} pool options, {fit['n_expert']} experts"
            print(f"    #{fit_index}: {fit['result']} (floor={fit['min_ctx']}, {counts}, logged split/context probes={len(fit['rpc_fracs'])}/{len(fit['grow'])})")
            if fit["selected"]:
                selected = fit["selected"]
                print(f"      selected: {selected['candidate']} n_ctx={selected['n_ctx']} rpc_frac={selected['rpc_frac']:.4f} split_size={selected['split_size']}")
            elif fit["grow"]:
                probe = fit["grow"][-1]
                budget = "" if probe["probes_used"] is None else f" probes={probe['probes_used']}/{probe['probes_limit']}"
                print(f"      latest context probe: n_ctx={probe['n_ctx']} {probe['verdict']}{budget}")
            if fit["rpc_fracs"]:
                probe = fit["rpc_fracs"][-1]
                print(f"      latest split probe: rpc_frac={probe['frac']:.4f} {probe['verdict']} (measure={probe['measured']}, devices={probe['devices']}, margins={probe['margin_ok']})")
            if fit["rejected_candidates"]:
                print(f"      min-ctx rejects: {', '.join(fit['rejected_candidates'])}")
            if fit["result"] == "in progress" and not fit["rpc_fracs"] and not fit["grow"]:
                print("      probe detail: no split/context probe lines found in this log")
    if r["fit_margins"]:
        print("  fit margins:")
        for device, name, target, margin in r["fit_margins"]:
            print(f"    device[{device}] {name}: fit-target={target} MiB base-margin={margin} MiB")
    if r["cache_events"]:
        print("  cache/feedback events:")
        for event in r["cache_events"]:
            print(f"    {event}")
    else:
        print("  cache/feedback events: none logged")
    if r["kv"] or r["rs"]:
        suffix = " (last observed; may be a fit probe)" if r["served_n_ctx"] is None else ""
        print(f"  KV buffer : {', '.join(f'{d}={v:.0f}' for d, v in sorted(r['kv'].items()))}{suffix}")
    if r["rs"]:
        suffix = " (last observed; may be a fit probe)" if r["served_n_ctx"] is None else ""
        print(f"  RS buffer : {', '.join(f'{d}={v:.0f}' for d, v in sorted(r['rs'].items()))}{suffix}")
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
