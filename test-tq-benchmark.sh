#!/bin/bash
# Qwen3.8-27B TurboQuant Benchmark Suite
# Automates quantization + inference benchmarking across TQ and IQ types
#
# Hardware: AMD RDNA4 16GB (ROCm0) + RTX 3060 Ti 8GB RPC at 192.168.5.2:50052
# Goal: 180K-200K context on 16GB GPU with best quality/speed tradeoff
#
# Usage:
#   ./test-tq-benchmark.sh --quantize    # Quantize missing models
#   ./test-tq-benchmark.sh --benchmark   # Run inference benchmarks
#   ./test-tq-benchmark.sh --both        # Quantize + benchmark
#   ./test-tq-benchmark.sh --quick       # Quick smoke test (existing models only)

set -euo pipefail

# === CONFIG ===
BUILD="$HOME/projects/projects2/llamacpp/llama-cpp-turboquant/build"
BIN="$BUILD/bin"
SERVER="$BIN/llama-server"
QUANT="$BIN/llama-quantize"
PERPLEXITY="$BIN/llama-perplexity"

BF16_1="/mnt/seagate/ai/Qwen3.8-27B-BF16-00001-of-00002.gguf"
BF16_2="/mnt/seagate/ai/Qwen3.8-27B-BF16-00002-of-00002.gguf"
OUT_DIR="/mnt/seagate/ai"
RPC="192.168.5.2:50052"
PORT=9080
INI="/mnt/ai/models/llama-router.ini"

# Model definitions: name -> (type, source_path)
# type: "existing" = already on disk, "TQ3_1S" etc = quantize from BF16
declare -A MODELS=(
    # Existing models
    ["Qwen3.8-27B-IQ4_XS"]="existing|/mnt/seagate/ai/Qwen3.8-27B-IQ4_XS.gguf|/mnt/seagate/ai/Qwen3.8-27B-IQ4_XS.gguf"
    ["Qwen3.8-27B-UD-IQ3_XXS"]="existing|/mnt/seagate/ai/Qwen3.8-27B-UD-IQ3_XXS.gguf|/mnt/seagate/ai/Qwen3.8-27B-UD-IQ3_XXS.gguf"
    ["Qwen3.8-27B-IQ2_S"]="existing|/mnt/ai/models/lmstudio/unsloth/Qwen3.8-27B/Qwen3.8-27B-UD-IQ2_S.gguf|/mnt/ai/models/lmstudio/unsloth/Qwen3.8-27B/Qwen3.8-27B-UD-IQ2_S.gguf"
    # New quantizations from BF16 (split files)
    ["Qwen3.8-27B-TQ3_1S"]="TQ3_1S|$BF16_1|/mnt/seagate/ai/Qwen3.8-27B-TQ3_1S"
    ["Qwen3.8-27B-TQ4_1S"]="TQ4_1S|$BF16_1|/mnt/seagate/ai/Qwen3.8-27B-TQ4_1S"
    # DFlash2 drafter re-quantized to TQ3_1S (standalone, single file)
    ["Qwen3.8-27B-DFlash2-TQ3_1S"]="TQ3_1S|/mnt/seagate/ai/Qwen3.8-27B-DFlash2-Q4_K_M.gguf|/mnt/seagate/ai/Qwen3.8-27B-DFlash2-TQ3_1S"
)

# MTP draft model (blk.64-only, 1.3GB) - for spec-type=draft-mtp tests
MTP_DRAFT="/mnt/ai/models/unsloth/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
# DFlash2 standalone drafter (blk.0-74, 1.1GB Q4_K_M) - for spec-type=draft-dflash tests
DFLASH_DRAFT="/mnt/seagate/ai/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
# DFlash2-TQ3_1S drafter (918MB, 4.0 BPW) - for spec-type=draft-dflash tests
DFLASH_TQ3_DRAFT="/mnt/seagate/ai/Qwen3.8-27B-DFlash2-TQ3_1S.gguf"

# KV cache types to test
KV_TYPES=("turbo3" "turbo4")

# Context sizes to sweep (tokens) - NO_MTP can go to 200K+
CTX_SIZES_NO_MTP=(32768 65536 98304 131072 180000 200000 220000)
# WITH_MTP is limited to ~75K-96K by MTP KV overhead
CTX_SIZES_WITH_MTP=(32768 65536 75000)

# Standard server env
export TURBO_AUTO_ASYMMETRIC=0
export GGML_TQ_NATIVE=1

# Test prompt (short, for smoke tests)
SMOKE_PROMPT="The theory of relativity by Albert Einstein states that"

# Long prompt for context sweep (fills KV cache)
FILL_TEXT="/tmp/llm-bench-fill.txt"
if [ ! -f "$FILL_TEXT" ]; then
    python3 -c "
import random
words = ['neural', 'network', 'deep', 'learning', 'model', 'attention',
         'transformer', 'language', 'model', 'quantum', 'computing',
         'physics', 'einstein', 'relativity', 'space', 'time', 'gravity',
         'biology', 'genetics', 'evolution', 'species', 'climate']
text = ' '.join(random.choices(words, k=8000)) * 4
with open('$FILL_TEXT', 'w') as f:
    f.write(text)
"
fi

# Benchmark INI - generated per-run for the target model
BENCH_INI="/tmp/llm-bench-$(date +%s).ini"
trap "rm -f $BENCH_INI" EXIT

# === HELPERS ===

log() { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARN: $*" >&2; }

check_server() {
    curl -s --max-time 2 "http://localhost:$PORT/health" > /dev/null 2>&1
}

wait_server() {
    local timeout=${1:-30}
    local count=0
    while ! check_server; do
        sleep 1
        ((count++)) || true
        if [ $count -ge $timeout ]; then
            echo "ERROR: Server failed to start within ${timeout}s"
            return 1
        fi
    done
}

stop_server() {
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 1
}

# === QUANTIZATION ===

quantize_if_needed() {
    local name=$1
    local type=$2
    local src=$3
    local out_base=$4

    if [ "$type" = "existing" ]; then
        if [ -f "$src" ]; then
            log "SKIP: $name exists at $src"
            return 0
        else
            warn "MISSING: $name at $src"
            return 1
        fi
    fi

    # Resolve actual source file(s) — append .gguf if not already present
    local src_shard1=""
    local src_shard2=""
    local out_shard1=""
    local is_split=false

    if [[ "$src" == *"-00001-of-"* ]]; then
        is_split=true
        src_shard1="$src"
        src_shard2="${src/00001/00002}"
    elif [[ "$src" == *"-00002-of-"* ]]; then
        is_split=true
        src_shard2="$src"
        src_shard1="${src/00002/00001}"
    elif [ -f "${src}.gguf" ]; then
        src_shard1="${src}.gguf"
    elif [ -f "$src" ]; then
        src_shard1="$src"
    else
        warn "MISSING: $name source not found (tried: ${src}, ${src}.gguf)"
        return 1
    fi

    out_shard1="${out_base}.gguf"

    # Check if output already exists
    if [ -f "$out_shard1" ]; then
        log "SKIP: $name already quantized at $out_shard1"
        return 0
    fi

    log "QUANTIZE: $name ($type) from $src -> $out_base"
    local size_before=$(du -sh "$src" | cut -f1)

    local max_retries=2
    local attempt=0
    local success=false

    while [ $attempt -le $max_retries ] && [ "$success" = "false" ]; do
        attempt=$((attempt + 1))
        if [ $attempt -gt 1 ]; then
            log "RETRY $attempt/$max_retries: $name ($type)"
        fi

        if timeout 3600 $QUANT --allow-requantize "$src_shard1" "$out_shard1" $type 2>&1 | tee -a /tmp/tq-quant.log; then
            success=true
        else
            exit_code=${PIPESTATUS[0]}
            if [ $attempt -lt $((max_retries + 1)) ]; then
                log "WARN: $name quantize attempt $attempt failed (exit $exit_code), retrying..."
                sleep 5
            else
                warn "FAIL: $name quantize failed after $((attempt)) attempts"
                return 1
            fi
        fi
    done

    local size_after=$(du -sh "$out_shard1" | cut -f1)
    log "DONE: $name quantized -> $size_after"
}

run_quantize() {
    log "=== QUANTIZATION PHASE ==="

    for model_name in "${!MODELS[@]}"; do
        IFS='|' read -r type src_path dst_path <<< "${MODELS[$model_name]}"
        quantize_if_needed "$model_name" "$type" "$src_path" "$dst_path" || true
    done

    log "=== QUANTIZATION COMPLETE ==="
}

# === SERVER LAUNCH ===

launch_server() {
    local model_path=$1
    local model_name=$2
    local kv_type=${3:-"turbo4"}
    local drafter_type=${4:-"none"}
    local extra_args=${5:-""}

    stop_server

    local safe_name="${model_name//\//-}"
    local log_file="/tmp/llm-server-${safe_name}.log"

    log "LAUNCH: $model_name (KV=$kv_type, drafter=$drafter_type) -> $log_file"

    cmd="TURBO_AUTO_ASYMMETRIC=0 GGML_TQ_NATIVE=1 $SERVER"
    cmd+=" -m '$model_path'"
    cmd+=" --port $PORT"
    cmd+=" --host 0.0.0.0"
    cmd+=" --rpc $RPC"
    cmd+=" -ctk $kv_type -ctv $kv_type"
    cmd+=" -fa on"
    cmd+=" -kvu"
    cmd+=" --parallel 1"
    cmd+=" -lv 4"
    cmd+=" --sleep-idle-seconds 7200"

    if [ "$drafter_type" = "mtp" ] && [ -f "$MTP_DRAFT" ]; then
        cmd+=" --spec-type draft-mtp"
        cmd+=" --spec-draft-model '$MTP_DRAFT'"
        cmd+=" --spec-draft-n-max 3"
    elif [ "$drafter_type" = "dflash2" ] && [ -f "$DFLASH_DRAFT" ]; then
        cmd+=" --spec-type draft-dflash"
        cmd+=" --spec-draft-model '$DFLASH_DRAFT'"
        cmd+=" --spec-draft-n-max 4"
        cmd+=" --spec-draft-p-min 0.1"
    elif [ "$drafter_type" = "dflash2-tq3" ] && [ -f "$DFLASH_TQ3_DRAFT" ]; then
        cmd+=" --spec-type draft-dflash"
        cmd+=" --spec-draft-model '$DFLASH_TQ3_DRAFT'"
        cmd+=" --spec-draft-n-max 4"
        cmd+=" --spec-draft-p-min 0.1"
    fi

    cmd+=" $extra_args"
    log "CMD: $cmd"

    eval "$cmd" > "$log_file" 2>&1 &
    echo $! > "/tmp/llm-server-$PORT.pid"

    wait_server 60 || {
        warn "Server failed to start for $model_name"
        tail -30 "$log_file"
        return 1
    }

    log "READY: $model_name (PID $(cat /tmp/llm-server-$PORT.pid))"
    return 0
}

# === INFERENCE BENCHMARK ===

benchmark_smoke() {
    local model_path=$1
    local model_name=$2
    local kv_type=${3:-"turbo4"}
    local drafter_type=${4:-"none"}

    log "SMOKE: $model_name (KV=$kv_type, drafter=$drafter_type)"

    launch_server "$model_path" "$model_name" "$kv_type" "$drafter_type" || return 1

    # Run smoke prompt - measure generation speed
    local result=$(curl -s --max-time 60 -X POST "http://localhost:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$SMOKE_PROMPT\", \"n_predict\": 32, \"stream\": false}" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tokens_per_second',0))" 2>/dev/null || echo "0")

    # Measure PP (prompt eval) speed
    local pp_result=$(curl -s --max-time 120 -X POST "http://localhost:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$(python3 -c "print('word ' * 5000)")\", \"n_predict\": 1, \"stream\": false}" 2>/dev/null)
    local pp_tps=$(echo "$pp_result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tokens_per_second',0))" 2>/dev/null || echo "0")

    log "SMOKE: $model_name -> TG=${result} tok/s PP=${pp_tps} tok/s"

    stop_server
    echo "$model_name|smoke|${kv_type}|drafter=${drafter_type}|TG=${result}|PP=${pp_tps}"
}

benchmark_context_sweep() {
    local model_path=$1
    local model_name=$2
    local kv_type=${3:-"turbo4"}
    local drafter_type=${4:-"none"}

    if [ "$drafter_type" = "mtp" ]; then
        local ctx_sizes=("${CTX_SIZES_WITH_MTP[@]}")
    else
        local ctx_sizes=("${CTX_SIZES_NO_MTP[@]}")
    fi

    log "CTX-SWEEP: $model_name (KV=$kv_type, drafter=$drafter_type)"

    launch_server "$model_path" "$model_name" "$kv_type" "$drafter_type" || return 1

    local safe_name="${model_name//\//-}"
    local server_log="/tmp/llm-server-${safe_name}.log"
    local fill_json="/tmp/llm-bench-fill-${safe_name}.json"
    local results=""
    local max_achieved=0

    for ctx in "${ctx_sizes[@]}"; do
        # Set context size
        curl -s -X POST "http://localhost:$PORT/options" \
            -H "Content-Type: application/json" \
            -d "{\"ctx_size\": $ctx}" > /dev/null 2>&1

        # Read fill text and write as proper JSON to temp file
        python3 -c "
import json, sys
text = open('$FILL_TEXT').read()
# Take first ctx*4 bytes for the prompt
taken = text[:int($ctx) * 4]
with open('$fill_json', 'w') as f:
    json.dump({'prompt': taken, 'n_predict': 1, 'stream': False}, f)
" 2>/dev/null

        # Fill cache using temp JSON file
        local fill_result=$(curl -s --max-time 120 \
            -X POST "http://localhost:$PORT/completion" \
            -H "Content-Type: application/json" \
            --data-binary "@$fill_json" 2>/dev/null)

        # Check for OOM in server log
        if grep -qi "memory\|OOM\|allocation failed\|out of memory" "$server_log" 2>/dev/null; then
            log "  ctx=$ctx -> OOM"
            break
        fi

        local kv_size=$(curl -s "http://localhost:$PORT/health" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('kv_cache_size',0))" 2>/dev/null || echo "0")

        # Measure eval speed
        local eval_result=$(curl -s --max-time 60 -X POST "http://localhost:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt": "Speed test sentence.", "n_predict": 64, "stream": false}' 2>/dev/null)
        local eval_tok_s=$(echo "$eval_result" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('tokens_per_second',0))" 2>/dev/null || echo "0")

        max_achieved=$ctx
        log "  ctx=$ctx kv=${kv_size}MiB eval=${eval_tok_s} tok/s"
        results+="ctx=${ctx}|kv=${kv_size}|eval=${eval_tok_s} "

        rm -f "$fill_json"
    done

    rm -f "$fill_json"
    stop_server
    echo "$model_name|ctx-sweep|${kv_type}|drafter=${drafter_type}|max_ctx=${max_achieved}|${results}"
}

run_benchmark() {
    log "=== BENCHMARK PHASE ==="

    local results_file="/tmp/llm-bench-results-$(date +%Y%m%d-%H%M%S).txt"
    echo "# Qwen3.8-27B TurboQuant Benchmark Results" > "$results_file"
    echo "# Date: $(date)" >> "$results_file"
    echo "# Hardware: AMD RDNA4 16GB + RTX 3060 Ti 8GB RPC at $RPC" >> "$results_file"
    echo "# Env: TURBO_AUTO_ASYMMETRIC=0 GGML_TQ_NATIVE=1" >> "$results_file"
    echo "# MTP draft: $MTP_DRAFT" >> "$results_file"
    echo "# DFlash2 draft (Q4_K_M): $DFLASH_DRAFT" >> "$results_file"
    echo "# DFlash2 draft (TQ3_1S): $DFLASH_TQ3_DRAFT" >> "$results_file"
    echo "" >> "$results_file"

    # DFlash2 drafter is tested with EVERY model (it's a standalone drafter)
    # MTP drafter only works if the model has embedded blk.64 (IQ4_XS, UD-IQ3_XXS, BF16-derived)
    # IQ2_S has no blk.64 - only DFlash2 or no-drafter applicable

    for model_name in "${!MODELS[@]}"; do
        IFS='|' read -r type src_path dst_path <<< "${MODELS[$model_name]}"

        # Use dst_path (the output / actual model path)
        local model_path="$dst_path"
        if [ -f "${dst_path}.gguf" ]; then
            model_path="${dst_path}.gguf"
        elif [ -f "${dst_path}-00001-of-00002.gguf" ]; then
            model_path="${dst_path}-00001-of-00002.gguf"
        fi

        # Skip missing existing models
        if [ "$type" = "existing" ] && [ ! -f "$model_path" ]; then
            warn "SKIP: $model_name missing at $model_path"
            continue
        fi

        if [ ! -f "$model_path" ]; then
            warn "SKIP: $model_name file not found at $model_path"
            continue
        fi

        # Determine which drafters work with this model
        local has_blk64=false
        $BIN/llama-quantize --dry-run "$model_path" /dev/null COPY 2>&1 | \
            grep -q 'blk\.64\.' && has_blk64=true

        local model_size=$(du -sh "$model_path" 2>/dev/null | cut -f1)

        for kv in "${KV_TYPES[@]}"; do
            # 1. No drafter (pure decode)
            log "--- $model_name KV=$kv no-drafter ---"
            local r=$(benchmark_smoke "$model_path" "$model_name" "$kv" "none" 2>&1)
            echo "$r" >> "$results_file"; log "$r"

            local cr=$(benchmark_context_sweep "$model_path" "$model_name" "$kv" "none" 2>&1)
            echo "$cr" >> "$results_file"; log "$cr"

            # 2. DFlash2 drafter (Q4_K_M, standalone, works with any main model)
            if [ -f "$DFLASH_DRAFT" ]; then
                log "--- $model_name KV=$kv dflash2 ---"
                local r2=$(benchmark_smoke "$model_path" "$model_name" "$kv" "dflash2" 2>&1)
                echo "$r2" >> "$results_file"; log "$r2"
            fi

            # 3. DFlash2-TQ3_1S drafter (918MB, 15% smaller than Q4_K_M)
            if [ -f "$DFLASH_TQ3_DRAFT" ]; then
                log "--- $model_name KV=$kv dflash2-tq3 ---"
                local r4=$(benchmark_smoke "$model_path" "$model_name" "$kv" "dflash2-tq3" 2>&1)
                echo "$r4" >> "$results_file"; log "$r4"
            fi

            # 4. MTP drafter (only if model has blk.64 embedded)
            if [ "$has_blk64" = "true" ] && [ -f "$MTP_DRAFT" ]; then
                log "--- $model_name KV=$kv mtp ---"
                local r3=$(benchmark_smoke "$model_path" "$model_name" "$kv" "mtp" 2>&1)
                echo "$r3" >> "$results_file"; log "$r3"
            fi
        done

        log "DONE: $model_name ($model_size)"
    done

    log "=== BENCHMARK COMPLETE ==="
    log "Results: $results_file"
    echo ""
    echo "=== SUMMARY ==="
    column -t -s '|' "$results_file" | tail -30
}

# === PERPLEXITY (if llama-perplexity available) ===

run_perplexity() {
    local model_path=$1
    local model_name=$2
    local test_file=${3:-"/tmp/llm-bench-test.txt"}

    if [ ! -f "$test_file" ]; then
        # Download a small test corpus
        python3 -c "
        import urllib.request
        # Use wikitext-2 as a quick perplexity test
        url = 'https://raw.githubusercontent.com/microsoft/Phi-3Cookbook/main/data/wikitext-2-test.txt'
        urllib.request.urlretrieve(url, '$test_file')
        " 2>/dev/null || echo "test sentence one two three four five" > "$test_file"
    fi

    log "PERPLEXITY: $model_name"

    # Note: perplexity needs a running server with the model loaded
    # Use the REST API if available, otherwise skip
    curl -s --max-time 60 -X POST "http://localhost:$PORT/perplexity" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$(cat $test_file)\"}" 2>/dev/null || \
    log "PERPLEXITY: $model_name skipped (API not available)"
}

# === MAIN ===

usage() {
    echo "Usage: $0 [--quantize|--benchmark|--both|--quick]"
    echo ""
    echo "  --quantize   Quantize missing models from BF16"
    echo "  --benchmark  Run inference benchmarks"
    echo "  --both       Quantize + benchmark (full pipeline)"
    echo "  --quick     Quick smoke test (existing models only)"
    echo ""
    echo "=== MODEL INVENTORY ==="
    echo ""
    echo "NEEDS QUANTIZATION:"
    echo "  Qwen3.8-27B-TQ3_1S  ~13.5 GB (4.13 BPW) - from BF16 split"
    echo "  Qwen3.8-27B-TQ4_1S  ~16.2 GB (5.08 BPW) - from BF16 split"
    echo "  Qwen3.8-27B-DFlash2-TQ3_1S ~918 MB (4.00 BPW) - from DFlash2-Q4_K_M, 15% smaller"
    echo ""
    echo "EXISTING MODELS:"
    echo "  Qwen3.8-27B-IQ4_XS        15 GB  IQ4_XS    [has blk.64 MTP, MTP or DFlash2 drafter]"
    echo "  Qwen3.8-27B-UD-IQ3_XXS  11 GB  UD-IQ3_XXS [has blk.64 MTP, MTP or DFlash2 drafter]"
    echo "  Qwen3.8-27B-IQ2_S         7.8 GB  IQ2       [NO blk.64, DFlash2 drafter only]"
    echo ""
    echo "DRAFT MODELS:"
    echo "  MTP blk.64-only:       $MTP_DRAFT"
    echo "  DFlash2 (Q4_K_M):     $DFLASH_DRAFT"
    echo "  DFlash2 (TQ3_1S):     $DFLASH_TQ3_DRAFT"
    echo ""
    echo "NOTE: Qwen3.8-27B is DENSE (not MoE). UD = Universal Distillation (Unsloth)."
    echo "      IQ2_S has NO embedded draft. All others have blk.64 self-MTP."
    echo ""
    echo "DRAFTER MATRIX:"
    echo "  Model              | no-drafter | MTP | DFlash2 | DFlash2-TQ3_1S"
    echo "  -------------------|------------|-----|---------|----------------"
    echo "  IQ4_XS            |     ok     |  ok |    ok   |  ok"
    echo "  UD-IQ3_XXS        |     ok     |  ok |    ok   |  ok"
    echo "  IQ2_S             |     ok     |  no |    ok   |  ok"
    echo "  TQ3_1S (new)      |     ok     |  ok |    ok   |  ok"
    echo "  TQ4_1S (new)      |     ok     |  ok |    ok   |  ok"
}

main() {
    local mode=${1:-"--both"}

    case "$mode" in
        --quantize)
            run_quantize
            ;;
        --benchmark)
            run_benchmark
            ;;
        --both)
            run_quantize
            run_benchmark
            ;;
        --quick)
            # Only existing models
            log "QUICK: Testing existing models only"
            KV_TYPES=("turbo4")
            run_benchmark
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
