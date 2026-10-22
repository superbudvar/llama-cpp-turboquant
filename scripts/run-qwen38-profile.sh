#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="$repo_root/build/bin/llama-server"
preset="/mnt/ai/models/llama-router.ini"
run_id="$(date +%Y%m%d-%H%M%S)"
log_file="/tmp/load8-$run_id.log"
profile_file="/tmp/opencode/qwen38-profile-$run_id.jsonl"

if [[ ! -x "$server" ]]; then
    printf 'llama-server is missing or not executable: %s\n' "$server" >&2
    exit 1
fi
if [[ ! -r "$preset" ]]; then
    printf 'model preset file is not readable: %s\n' "$preset" >&2
    exit 1
fi

mkdir -p /tmp/opencode
printf 'server log: %s\nactivation profile: %s\n' "$log_file" "$profile_file"

env \
    TURBO_AUTO_ASYMMETRIC=1 \
    GGML_TQ_NATIVE=1 \
    ${GGML_TQ_MMQ:+GGML_TQ_MMQ="$GGML_TQ_MMQ"} \
    LLAMA_ACTIVATION_PROFILE="$profile_file" \
    "$server" \
    --models-preset "$preset" \
    --port 9080 \
    --host 0.0.0.0 \
    --models-max 1 \
    -lv 4 \
    --sleep-idle-seconds 7200 \
    2>&1 | tee "$log_file"
