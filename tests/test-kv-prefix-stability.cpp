// I0 probe for the chunked disk format pre-mortem (pm-20260916-301).
//
// The chunk store content-addresses KV state chunks by bytes of the
// llama_state_seq_get_data output. That only works if the SAME token prefix
// serializes to byte-identical state regardless of HOW it was computed:
//   - decode batch shape (one big ubatch vs 1-token-at-a-time)
//   - context geometry (n_ctx, n_parallel, cache types)
//   - history (append-continuation vs cold prefill)
//
// This test builds the same prompt in several independent contexts using
// different decode schedules, serializes seq 0 state from each, and compares
// byte-for-byte. A divergence means content-addressing silently stores
// duplicates (dedup = 0) and the chunk store must become a non-dedup append
// store, as the judges predicted.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <cinttypes>
#include <clocale>
#include <random>
#include <vector>

static llama_tokens load_real_tokens(const char * path) {
    llama_tokens tokens;
    FILE * f = fopen(path, "r");
    if (!f) {
        return tokens;
    }
    int32_t t;
    while (fscanf(f, "%d", &t) == 1) {
        tokens.push_back((llama_token) t);
    }
    fclose(f);
    return tokens;
}

struct probe_ctx {
    llama_context_ptr ctx;
    std::vector<uint8_t> state;
    std::string label;
};

// build a context with a given geometry and decode schedule, then dump seq state.
// schedule is a list of (offset, nb) pairs; nb=0 means one-at-a-time for that range,
// while the next step starts after the previous range. ranges are contiguous.
struct sched_step { size_t offset; size_t n; };

static probe_ctx build_and_serialize(
        llama_model                * model,
        const llama_tokens         & tokens,
        const llama_context_params & cparams,
        const std::vector<sched_step> & schedule,
        const std::string         & label,
        llama_state_seq_flags       flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    probe_ctx p;
    p.label = label;

    p.ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    if (!p.ctx) {
        LOG_ERR("%s: failed to init context for %s\n", __func__, label.c_str());
        return {};
    }

    size_t cur = 0;
    // the schedule's last step decides how many tokens are decoded; a trailing
    // 1-at-a-time step runs to the end, a bounded step stops at its offset+n
    size_t covered = 0;
    for (const auto & st : schedule) {
        covered = (st.n == 0) ? tokens.size() : std::min(tokens.size(), st.offset + st.n);
    }
    for (const auto & st : schedule) {
        (void) st;
        GGML_ASSERT(cur == st.offset);
        if (st.n == 0) {
            // one token at a time
            llama_batch batch = llama_batch_init(1, 0, 1);
            for (size_t i = cur; i < tokens.size(); ++i) {
                batch.n_tokens = 1;
                batch.token[0] = tokens[i];
                batch.pos[0]   = (llama_pos) i;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = true;
                if (llama_decode(p.ctx.get(), batch)) {
                    LOG_ERR("%s: failed to decode at %zu for %s\n", __func__, i, label.c_str());
                    llama_batch_free(batch);
                    return {};
                }
                cur++;
            }
            llama_batch_free(batch);
        } else {
            const size_t end = std::min(tokens.size(), cur + st.n);
            llama_batch batch = llama_batch_init((int32_t) (end - cur), 0, 1);
            for (size_t i = cur; i < end; ++i) {
                common_batch_add(batch, tokens[i], (llama_pos) i, {0}, i == end - 1);
            }
            if (llama_decode(p.ctx.get(), batch)) {
                LOG_ERR("%s: failed to decode at %zu for %s\n", __func__, cur, label.c_str());
                llama_batch_free(batch);
                return {};
            }
            llama_batch_free(batch);
            cur = end;
        }
    }
    GGML_ASSERT(cur == covered);

    const size_t size = llama_state_seq_get_size_ext(p.ctx.get(), 0, flags);
    if (size == 0) {
        LOG_ERR("%s: empty state for %s\n", __func__, label.c_str());
        return {};
    }
    p.state.resize(size);
    const size_t n = llama_state_seq_get_data_ext(p.ctx.get(), p.state.data(), p.state.size(), 0, flags);
    if (n != size) {
        LOG_ERR("%s: short serialize for %s (%zu != %zu)\n", __func__, label.c_str(), n, size);
        p.state.clear();
        return {};
    }

    const char * dump_dir = getenv("KV_PROBE_DUMP_DIR");
    if (dump_dir && *dump_dir) {
        std::string fn = std::string(dump_dir) + "/" + label + ".bin";
        FILE * f = fopen(fn.c_str(), "wb");
        if (f) {
            fwrite(p.state.data(), 1, p.state.size(), f);
            fclose(f);
            LOG("%s: dumped %zu bytes to %s\n", __func__, p.state.size(), fn.c_str());
        }
    }
    return p;
}

static bool compare(const probe_ctx & a, const probe_ctx & b, size_t * first_diff_out) {
    const auto & A = a.state;
    const auto & B = b.state;

    const size_t n = std::min(A.size(), B.size());
    size_t first_diff = n;
    size_t n_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        if (A[i] != B[i]) {
            if (n_diff == 0) {
                first_diff = i;
            }
            n_diff++;
        }
    }

    LOG("%-32s vs %-32s : %zu vs %zu bytes, %zu bytes differ over min=%zu, first at %zu\n",
            a.label.c_str(), b.label.c_str(), A.size(), B.size(), n_diff, n, first_diff);
    if (first_diff_out) {
        *first_diff_out = first_diff;
    }
    return n_diff == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx = 2048;
    params.kv_unified = true;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: no model file (-m)\n", __func__);
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params, true);
    llama_model * model = llama_init ? llama_init->model() : nullptr;
    if (!model) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    llama_tokens tokens;
    const char * tf = getenv("KV_PROBE_TOKENS_FILE");
    if (tf && *tf) {
        tokens = load_real_tokens(tf);
        LOG("%s: loaded %zu tokens from %s\n", __func__, tokens.size(), tf);
    } else if (params.prompt.empty()) {
        const auto * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < 512; ++i) {
            tokens.push_back(dist(rng));
        }
    } else {
        tokens = common_tokenize(llama_model_get_vocab(model), params.prompt, true, false);
    }
    if (tokens.empty()) {
        LOG_ERR("%s: no tokens\n", __func__);
        return 1;
    }
    if ((int32_t) tokens.size() > params.n_ctx) {
        tokens.resize(params.n_ctx);
    }

    LOG("%s: probing %zu tokens\n", __func__, tokens.size());

    auto cparams = common_context_params_to_llama(params);

    std::vector<sched_step> sched64;
    for (size_t i = 0; i < tokens.size(); i += 64) {
        sched64.push_back({i, 64});
    }

    probe_ctx whole_batch = build_and_serialize(model, tokens, cparams,
            {{0, tokens.size()}}, "whole-batch");
    probe_ctx ubatch64    = build_and_serialize(model, tokens, cparams, sched64, "ubatch=64");
    probe_ctx ubatch1     = build_and_serialize(model, tokens, cparams,
            {{0, 0}}, "token-at-a-time");

    // mixed: the realistic server pattern. first half batched (prefill),
    // second half one-at-a-time (generation). rows 0..N/2 should match the
    // pure-batch run byte-for-byte iff row bytes only depend on the token
    // and the prefill schedule.
    probe_ctx mixed = build_and_serialize(model, tokens, cparams,
            {{0, tokens.size() / 2}, {tokens.size() / 2, 0}}, "prefill+generate");

    if (whole_batch.state.empty() || ubatch64.state.empty() || ubatch1.state.empty() || mixed.state.empty()) {
        LOG_ERR("%s: failed to build contexts\n", __func__);
        return 1;
    }

    // gate (2) checkpoint-first restore: the server stores checkpoints with
    // LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY. test whether a PARTIAL_ONLY blob
    // restored into a fresh context + a small batched delta reprocess equals
    // the full whole-batch blob byte-for-byte, and how large the partial blob
    // is relative to the full one. this decides whether restoring from the
    // .ccn spill can substitute for the full 6.6 GiB .dpc read.
    {
        LOG("\n=== checkpoint-restore scenario (PARTIAL_ONLY) ===\n");

        const size_t split = tokens.size() / 2; // checkpoint covers first half, delta = second half

        probe_ctx ckpt = build_and_serialize(model, tokens, cparams,
                {{0, split}}, "ckpt-blob",
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (ckpt.state.empty()) {
            LOG_ERR("%s: failed to build checkpoint blob\n", __func__);
            return 1;
        }

        LOG("%-32s vs %-32s : checkpoint full-window ratio = %.3f\n",
                "ckpt-blob", "whole-batch", (double) ckpt.state.size() / (double) whole_batch.state.size());

        // restore the partial blob into a fresh context, then reprocess the
        // second half batched (the "delta")
        probe_ctx ckpt_delta;
        ckpt_delta.label = "ckpt+delta (full-window after)";
        ckpt_delta.ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
        if (!ckpt_delta.ctx) {
            LOG_ERR("%s: failed to init restore context\n", __func__);
            return 1;
        }
        const size_t n_set = llama_state_seq_set_data_ext(ckpt_delta.ctx.get(), ckpt.state.data(), ckpt.state.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n_set != ckpt.state.size()) {
            LOG_ERR("%s: partial restore short (%zu != %zu)\n", __func__, n_set, ckpt.state.size());
            return 1;
        }

        llama_batch batch = llama_batch_init((int32_t) (tokens.size() - split), 0, 1);
        for (size_t i = split; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], (llama_pos) i, {0}, i == tokens.size() - 1);
        }
        if (llama_decode(ckpt_delta.ctx.get(), batch)) {
            LOG_ERR("%s: failed to decode delta\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);

        const size_t sz = llama_state_seq_get_size_ext(ckpt_delta.ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        ckpt_delta.state.resize(sz);
        const size_t n_out = llama_state_seq_get_data_ext(ckpt_delta.ctx.get(), ckpt_delta.state.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_out != sz) {
            LOG_ERR("%s: short serialize after delta (%zu != %zu)\n", __func__, n_out, sz);
            return 1;
        }

        size_t d_ckpt = 0;
        const bool ok_ckpt = compare(whole_batch, ckpt_delta, &d_ckpt);
        LOG("%s\n", ok_ckpt ? "checkpoint-first restore: PASS (restore+delta == full compute)"
                            : "checkpoint-first restore: FAIL (restore+delta != full compute)");
        LOG("\n");
        if (!ok_ckpt) {
            LOG_WRN("%s: checkpoint-first restore diverges at %zu - gate (2) blocked, see d_ckpt\n", __func__, d_ckpt);
            return 1;
        }
    }

    LOG("\n=== results ===\n");

    size_t d01 = 0, d02 = 0, d12 = 0, dm0 = 0;
    const bool ok01 = compare(whole_batch, ubatch64, &d01);
    const bool ok02 = compare(whole_batch, ubatch1,  &d02);
    const bool ok12 = compare(ubatch64,    ubatch1,  &d12);
    const bool okm0 = compare(mixed,       ubatch64, &dm0);
    LOG("\n");

    const size_t half = mixed.state.size() * 2 / 3; // report rough proportion of diverging bytes later
    (void) half;
    const bool ok = ok01 && ok02 && ok12 && okm0;
    if (ok) {
        LOG("%s: PASS - all decode schedules produced byte-identical seq states\n", __func__);
    } else {
        LOG("%s: FAIL - independent computations of the same prefix diverge (diffs at %zu / %zu / %zu / %zu)\n",
                __func__, d01, d02, d12, dm0);
    }
    return ok ? 0 : 1;
}