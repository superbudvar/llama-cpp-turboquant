#!/bin/bash
# Qwen3.8-27B Benchmark Runner  — API-only, no server management
# Server runs ONCE in router mode; script calls API to select model
#
# SERVER (one-time setup, run by you):
#   TURBO_AUTO_ASYMMETRIC=0 GGML_TQ_NATIVE=1 \
#   ~/projects/.../llama-server \
#     --models-preset ./bench-qwen38.ini \
#     --log-file /tmp/bench-router.log \
#     --port 9081 --host 0.0.0.0 \
#     --models-max 1 --sleep-idle-seconds 7200
#
# BENCHMARK (run by script):
#   ./bench-runner.sh --port 9081 [--resume]
#
# PRESETS available (see bench-qwen38.ini for full list):
#   IQ4_XS-no-draft-t4  IQ4_XS-dflash2-t4  IQ4_XS-mtp-t4
#   UD-IQ3_XXS-no-draft-t4  UD-IQ3_XXS-dflash2-t4  UD-IQ3_XXS-mtp-t4
#   IQ2_S-no-draft-t4  IQ2_S-dflash2-t4  IQ2_S-dflash2-tq3-t4
#   TQ3_1S-no-draft-t4  TQ3_1S-dflash2-t4  TQ3_1S-dflash2-tq3-t4
#   TQ4_1S-no-draft-t4  TQ4_1S-mtp-t4

set -euo pipefail

PORT="${PORT:-9081}"
HOST="${HOST:-localhost}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/llm-bench-results}"
SMOKE_TOKENS=32
EVAL_TOKENS=64

mkdir -p "$RESULTS_DIR"

usage() {
    echo "Usage: $0 [--port <port>] [--preset <name>] [--ctx-sizes <csv>] [--resume]"
    echo ""
    echo "  --port      Server port (default: 9081)"
    echo "  --preset    INI section name to benchmark (default: all)"
    echo "  --ctx-sizes Comma-separated context sizes (default: 32768,65536,98304,131072,180000)"
    echo "  --resume    Skip presets that already have result files"
    echo ""
    echo "Examples:"
    echo "  $0 --port 9081                           # run all presets"
    echo "  $0 --port 9081 --resume                 # run all, skip completed"
    echo "  $0 --port 9081 --preset IQ4_XS-no-draft-t4  # run one preset"
    echo ""
    echo "To list all presets:  grep '^\[' bench-qwen38.ini | tr -d '[]'"
}

list_presets() {
    echo "Available presets:"
    grep '^\[' bench-qwen38.ini | tr -d '[]' | grep -v '^\*$' | sort
    echo ""
    echo "Note: TQ3_1S/TQ4_1S presets require running --quantize first."
}

wait_ready() {
    local timeout=${1:-300}
    local count=0
    log "Waiting for server on port $PORT..."
    while true; do
        local status=$(curl -s --max-time 3 "http://${HOST}:${PORT}/health" 2>/dev/null | \
            python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "not_ready")
        if [ "$status" = "ok" ]; then
            log "Server ready"
            return 0
        fi
        sleep 3
        ((count++)) || true
        if [ $count -ge $timeout ]; then
            echo "ERROR: Server not ready after ${timeout}s"
            exit 1
        fi
    done
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Wait for a specific model to finish loading before issuing requests
wait_model_ready() {
    local model=$1
    local timeout=${2:-600}
    local count=0
    log "Waiting for model '$model' to finish loading..."
    while true; do
        local result=$(curl -s --max-time 15 -X POST "http://${HOST}:${PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"prompt\": \"test\", \"n_predict\": 1, \"stream\": false}" 2>/dev/null)
        if [ -n "$result" ] && echo "$result" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
            log "Model '$model' ready"
            sleep 2
            return 0
        fi
        sleep 10
        ((count++)) || true
        if [ $count -ge $timeout ]; then
            echo "ERROR: Model '$model' not ready after ${timeout}s"
            return 1
        fi
    done
}

# Parse tokens_per_second from completion response, trying multiple field names
parse_tps() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
# try timings.predicted_per_second (llama-server standard)
if 'timings' in d:
    t = d['timings']
    for key in ['predicted_per_second', 'tok_per_sec', 'tps']:
        if key in t:
            print(t[key]); sys.exit(0)
# fallback: top-level
for key in ['tokens_per_second', 'tok_per_sec', 'tps', 'generation_tokens_per_second']:
    if key in d:
        print(d[key]); sys.exit(0)
print('0')
" 2>/dev/null || echo "0"
}

# Get context size from server for a given model
get_ctx() {
    local model=$1
    curl -s --max-time 5 "http://${HOST}:${PORT}/props?model=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$model'))")" 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('llama_props',{}).get('context_length',0))" 2>/dev/null || echo "0"
}

# Run a single completion and return TPS (predicted_per_second)
run_completion() {
    local model=$1
    local prompt=$2
    local n_predict=$3
    local timeout=$4

    local payload=$(python3 -c "
import json
print(json.dumps({
    'model': '$model',
    'prompt': '''$prompt''',
    'n_predict': $n_predict,
    'stream': False
}))
" 2>/dev/null)

    local result=$(curl -s --max-time "$timeout" -X POST "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)

    parse_tps <<< "$result"
}

# Run a completion and return prompt processing speed (prompt_per_second)
run_pp() {
    local model=$1
    local prompt=$2
    local timeout=$3

    local payload=$(python3 -c "
import json
print(json.dumps({
    'model': '$model',
    'prompt': '''$prompt''',
    'n_predict': 32,
    'stream': False
}))
" 2>/dev/null)

    local result=$(curl -s --max-time "$timeout" -X POST "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)

    python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
for key in ['prompt_per_second', 'prompt_per_token_ms']:
    if key in t:
        print(t[key]); sys.exit(0)
print('0')
" <<< "$result" 2>/dev/null || echo "0"
}

# Smoke test: eval speed at a given context
smoke_test() {
    local model=$1
    local ctx_size=${2:-98304}

    log "  SMOKE ctx=${ctx_size}"

    # Wait for model to finish loading (handles model switch)
    wait_model_ready "$model" 300 || return 1

    # Set ctx via props
    curl -s --max-time 5 -X POST "http://${HOST}:${PORT}/props" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"ctx_size\": $ctx_size}" > /dev/null 2>&1 || true

    sleep 1

    # TG: short generation (retry on timeout)
    local tps=0
    for attempt in 1 2 3; do
        tps=$(run_completion "$model" "The theory of relativity states that" $SMOKE_TOKENS 300)
        if [ "$tps" != "0" ]; then
            break
        fi
        log "    TG timeout, retry $attempt/3..."
        sleep 10
    done
    log "    TG=${tps} tok/s"

    # PP: prompt processing speed on long prompt (retry on timeout)
    local prompt_len=$((ctx_size / 4))
    local pp_seed=$RANDOM
    local pp_prompt=$(python3 -c "
import random
random.seed($pp_seed)
words = 'neural network deep learning model attention transformer language quantum computing physics einstein relativity space time gravity biology genetics evolution climate data algorithm matrix'.split()
text = ' '.join(random.choices(words, k=12000))
print(text[:$prompt_len])
" 2>/dev/null)
    local pp_tps=0
    for attempt in 1 2 3; do
        pp_tps=$(run_pp "$model" "$pp_prompt" 600)
        if [ "$pp_tps" != "0" ]; then
            break
        fi
        log "    PP timeout, retry $attempt/3..."
        sleep 10
    done
    log "    PP=${pp_tps} tok/s"

    echo "smoke|ctx=${ctx_size}|TG=${tps}|PP=${pp_tps}"
}

# Context sweep: find max stable context
ctx_sweep() {
    local model=$1
    shift
    local ctx_sizes=("$@")

    log "  CTX-SWEEP"
    wait_model_ready "$model" 600 || return 1

    local max_achieved=0
    local results=""

    for ctx in "${ctx_sizes[@]}"; do
        # Set ctx
        curl -s --max-time 5 -X POST "http://${HOST}:${PORT}/props" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"ctx_size\": $ctx}" > /dev/null 2>&1 || true
        sleep 1

        # Fill cache with text proportional to ctx (unique seed per ctx to bust prompt cache)
        # Use 0.5x ctx to keep prefill time reasonable (~2-3 min at 400 tok/s)
        local fill_len=$((ctx / 2))
        local fill_seed=$RANDOM
        local fill_prompt=$(python3 -c "
import random
random.seed($fill_seed)
words = 'neural network deep learning model attention transformer language quantum computing physics einstein relativity space time gravity biology genetics evolution climate data algorithm matrix systems architecture performance optimization'.split()
text = ' '.join(random.choices(words, k=12000))
print(text[:$fill_len])
" 2>/dev/null)

        # Write prompt to temp file to avoid shell escaping hell
        local tmpfile="/tmp/bench-fill-${model//\//-}-${ctx}.json"
        python3 -c "
import json
with open('$tmpfile', 'w') as f:
    json.dump({
        'model': '$model',
        'prompt': '''$fill_prompt''',
        'n_predict': 1,
        'stream': False
    }, f)
" 2>/dev/null

        local fill_result=$(curl -s --max-time 1800 \
            -X POST "http://${HOST}:${PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            --data-binary "@$tmpfile" 2>/dev/null)
        rm -f "$tmpfile"

        # Check health for OOM/error
        local health=$(curl -s --max-time 5 "http://${HOST}:${PORT}/health?model=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$model'))")" 2>/dev/null)
        local has_error=$(echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); print('error' in str(d).lower())" 2>/dev/null || echo "false")

        if [ "$has_error" = "True" ]; then
            log "    ctx=${ctx} OOM/ERROR"
            break
        fi

        # KV cache size
        local kv_size=$(echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('kv_cache_size',0))" 2>/dev/null || echo "0")

        # Eval speed
        local eval_tps=$(run_completion "$model" "Speed test prompt." $EVAL_TOKENS 300)

        max_achieved=$ctx
        log "    ctx=${ctx} kv=${kv_size}MiB eval=${eval_tps} tok/s"
        results+="ctx=${ctx}|kv=${kv_size}|eval=${eval_tps} "
    done

    echo "ctx-sweep|max=${max_achieved}|${results}"
}

# Run all benchmarks for one preset
bench_preset() {
    local preset=$1
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local result_file="${RESULTS_DIR}/${preset//\//-}-${timestamp}.txt"

    log "=== PRESET: $preset ==="

    {
        echo "# Preset: $preset"
        echo "# Date: $(date)"
        echo "# Port: $PORT"
        echo ""

        # Get model info from server
        echo "# Available models on server:"
        curl -s "http://${HOST}:${PORT}/models" 2>/dev/null | python3 -m json.tool 2>/dev/null | grep '"id"' | head -20
        echo ""

        # Smoke test at 3 context sizes
        smoke_test "$preset" 32768
        smoke_test "$preset" 98304

        # Context sweep
        ctx_sweep "$preset" 32768 65536 98304 131072 180000

    } | tee "$result_file"

    log "DONE: $preset -> $result_file"
}

# Print section names from INI
get_all_presets() {
    grep '^\[' bench-qwen38.ini | tr -d '[]' | grep -v '^\*$' | sort
}

main() {
    local target_preset=""
    local ctx_sizes_str="32768,65536,98304,131072,180000"
    local resume=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port) PORT=$2; shift 2 ;;
            --preset) target_preset=$2; shift 2 ;;
            --ctx-sizes) ctx_sizes_str=$2; shift 2 ;;
            --resume) resume=true; shift ;;
            --list) list_presets; exit 0 ;;
            --help|-h) usage; exit 0 ;;
            *) echo "Unknown: $1"; usage; exit 1 ;;
        esac
    done

    wait_ready 60

    local timestamp=$(date +%Y%m%d-%H%M%S)

    if [ -n "$target_preset" ]; then
        bench_preset "$target_preset"
    else
        # Run all presets
        local all_file="${RESULTS_DIR}/all-presets-${timestamp}.txt"
        {
            echo "# All presets benchmark"
            echo "# Date: $(date)"
            echo "# Port: $PORT"
            echo ""

            for preset in $(get_all_presets); do
                if [ "$resume" = "true" ] && ls "${RESULTS_DIR}/${preset//\//-}"-*.txt &>/dev/null; then
                    log "SKIP: $preset already has results"
                    continue
                fi
                bench_preset "$preset"
                echo ""
            done
        } | tee "$all_file"
        log "ALL DONE -> $all_file"
    fi
}

main "$@"
