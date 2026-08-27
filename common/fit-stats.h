#pragma once

#include <cstdint>
#include <map>
#include <string>

// Post-load per-device overhead cache for the rpc-aware / maxctx fit.
//
// The fit's reserved-size breakdown omits the speculative draft model, extra
// graph/scratch buffers, and ROCm lazy-map internals. After a real warmup load
// the true RSS is known, so we record the gap (overhead = true_used - reserved)
// per backend device and apply it on later loads so the per-device fit margin
// is honored instead of being silently under-estimated.
//
// The stat is keyed by model path + device name + file mtime/size so a
// re-quantized model invalidates the old entry. When the model is recorded
// while another model shares the GPU, the overhead is inflated by the
// co-tenant's usage; this is conservative (the fit backs off more than needed)
// and self-corrects the next time the model is recorded alone.
namespace fit_stats {

// $LLAMA_FIT_STATS, else ~/.cache/llama-cpp/fit-stats.txt
std::string default_path();

// device-name -> overhead bytes for model_path (stale mtime/size ignored)
std::map<std::string, int64_t> load_overheads(
        const std::string & model_path,
        const std::string & path = default_path());

// record/update one device's overhead (model mtime/size captured for invalidation)
void record_overhead(
        const std::string & model_path,
        const std::string & dev_name,
        int64_t overhead_bytes,
        const std::string & path = default_path());

}  // namespace fit_stats
