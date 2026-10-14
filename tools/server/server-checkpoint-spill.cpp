#include "server-task.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// spill segment layout (all little-endian):
//   u32 magic, i64 n_tokens, i32 pos_min, i32 pos_max,
//   u64 len_tgt, u64 len_dft, u64 len_spec, tgt[len_tgt], dft[len_dft], spec[len_spec]
constexpr uint32_t CSPILL_MAGIC   = 0x4B505343; // "CSPK"
constexpr uint32_t CSPILL_HDR_SIZE = 4 + 8 + 4 + 4 + 8 + 8 + 8;

namespace {

bool cspill_check_header(const int64_t n_tokens, const llama_pos pos_min, const llama_pos pos_max, const uint64_t len_tgt, const uint64_t len_dft, const uint64_t len_spec) {
    if (n_tokens < 0 || pos_min < 0 || pos_max < 0) {
        return false;
    }
    if (n_tokens > (int64_t) (1ull << 40) || len_tgt > (1ull << 34) || len_dft > (1ull << 34) || len_spec > (1ull << 34)) {
        return false;
    }
    return len_tgt + len_dft + len_spec < (1ull << 36);
}

} // namespace

// these live at file scope (declared in server-task.h) so both the live
// slot and the async/RAM cache tiers can spill and rehydrate

int64_t server_ckpt_spill_store(common_prompt_checkpoint & ckpt, const std::string & path, bool truncate) {
    if (path.empty()) {
        return -1;
    }

    const size_t len_tgt  = ckpt.data_tgt.size();
    const size_t len_dft  = ckpt.data_dft.size();
    const size_t len_spec = ckpt.data_spec.size();

    FILE * f = fopen(path.c_str(), truncate ? "wb" : "ab");
    if (f == nullptr) {
        return -1;
    }

    const int64_t off = (int64_t) ftell(f);

    bool ok = fwrite(&CSPILL_MAGIC,   4,            1, f) == 1;
    ok = ok && fwrite(&ckpt.n_tokens, sizeof(int64_t), 1, f) == 1;
    ok = ok && fwrite(&ckpt.pos_min,  sizeof(llama_pos), 1, f) == 1;
    ok = ok && fwrite(&ckpt.pos_max,  sizeof(llama_pos), 1, f) == 1;

    const uint64_t len_tgt_u  = (uint64_t) len_tgt;
    const uint64_t len_dft_u  = (uint64_t) len_dft;
    const uint64_t len_spec_u = (uint64_t) len_spec;

    ok = ok && fwrite(&len_tgt_u,  sizeof(uint64_t), 1, f) == 1;
    ok = ok && fwrite(&len_dft_u,  sizeof(uint64_t), 1, f) == 1;
    ok = ok && fwrite(&len_spec_u, sizeof(uint64_t), 1, f) == 1;

    ok = ok && (len_tgt  == 0 || fwrite(ckpt.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
    ok = ok && (len_dft  == 0 || fwrite(ckpt.data_dft.data(),  1, len_dft,  f) == len_dft);
    ok = ok && (len_spec == 0 || fwrite(ckpt.data_spec.data(), 1, len_spec, f) == len_spec);

    const size_t seg = CSPILL_HDR_SIZE + len_tgt + len_dft + len_spec;

    if (fclose(f) != 0) {
        ok = false;
    }

    if (!ok) {
        return -1;
    }

    ckpt.spill_off = off;
    ckpt.spill_len = seg;

    ckpt.data_tgt.clear();
    ckpt.data_dft.clear();
    ckpt.data_spec.clear();

    return off;
}

bool server_ckpt_spill_load(common_prompt_checkpoint & ckpt, const std::string & path) {
    if (!ckpt.spilled()) {
        return true;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    bool ok = fseek(f, ckpt.spill_off, SEEK_SET) == 0;

    uint32_t magic = 0;
    ok = ok && fread(&magic, 4, 1, f) == 1;
    ok = ok && magic == CSPILL_MAGIC;

    std::vector<uint8_t> tgt, dft, spec;

    int64_t  n_tokens = 0;
    llama_pos pos_min = 0, pos_max = 0;
    uint64_t len_tgt = 0, len_dft = 0, len_spec = 0;

    ok = ok && fread(&n_tokens, sizeof(int64_t), 1, f) == 1;
    ok = ok && fread(&pos_min,   sizeof(llama_pos), 1, f) == 1;
    ok = ok && fread(&pos_max,   sizeof(llama_pos), 1, f) == 1;
    ok = ok && fread(&len_tgt,   sizeof(uint64_t), 1, f) == 1;
    ok = ok && fread(&len_dft,   sizeof(uint64_t), 1, f) == 1;
    ok = ok && fread(&len_spec,  sizeof(uint64_t), 1, f) == 1;

    ok = ok && cspill_check_header(n_tokens, pos_min, pos_max, len_tgt, len_dft, len_spec);

    // only accept data that matches the in-RAM metadata; anything else is a corrupt file
    ok = ok && n_tokens == ckpt.n_tokens && pos_min == ckpt.pos_min && pos_max == ckpt.pos_max;

    if (ok) {
        tgt.resize((size_t) len_tgt);
        dft.resize((size_t) len_dft);
        spec.resize((size_t) len_spec);
        ok = ok && (len_tgt  == 0 || fread(tgt.data(),  1, (size_t) len_tgt,  f) == len_tgt);
        ok = ok && (len_dft  == 0 || fread(dft.data(),  1, (size_t) len_dft,  f) == len_dft);
        ok = ok && (len_spec == 0 || fread(spec.data(), 1, (size_t) len_spec, f) == len_spec);
    }

    fclose(f);

    if (!ok) {
        return false;
    }

    ckpt.data_tgt = std::move(tgt);
    ckpt.data_dft = std::move(dft);
    ckpt.data_spec = std::move(spec);

    ckpt.spill_off = -1;
    ckpt.spill_len = 0;

    return true;
}

void server_ckpt_spill_compact(server_prompt & prompt, const std::string & path) {
    if (path.empty()) {
        return;
    }

    std::error_code ec;
    const uint64_t file_size = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        return;
    }

    if (prompt.checkpoints.empty()) {
        return;
    }

    // live payload: full checkpoints count their blobs, spilled ones their segment
    size_t live = 0;
    for (const auto & cur : prompt.checkpoints) {
        live += cur.size();
    }

    if ((uint64_t) live * 2 >= file_size) {
        return;
    }

    // rewrite with only the still-referenced segments
    const std::string tmp = path + ".tmp";

    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return;
    }

    int64_t off = 0;
    for (const auto & cur : prompt.checkpoints) {
        if (!cur.spilled()) {
            continue;
        }

        const size_t len_tgt  = cur.data_tgt.size();
        const size_t len_dft  = cur.data_dft.size();
        const size_t len_spec = cur.data_spec.size();

        bool ok = fwrite(&CSPILL_MAGIC,   4,              1, f) == 1;
        ok = ok && fwrite(&cur.n_tokens, sizeof(int64_t), 1, f) == 1;
        ok = ok && fwrite(&cur.pos_min,  sizeof(llama_pos), 1, f) == 1;
        ok = ok && fwrite(&cur.pos_max,  sizeof(llama_pos), 1, f) == 1;

        const uint64_t len_tgt_u  = (uint64_t) len_tgt;
        const uint64_t len_dft_u  = (uint64_t) len_dft;
        const uint64_t len_spec_u = (uint64_t) len_spec;

        ok = ok && fwrite(&len_tgt_u,  sizeof(uint64_t), 1, f) == 1;
        ok = ok && fwrite(&len_dft_u,  sizeof(uint64_t), 1, f) == 1;
        ok = ok && fwrite(&len_spec_u, sizeof(uint64_t), 1, f) == 1;

        ok = ok && (len_tgt  == 0 || fwrite(cur.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
        ok = ok && (len_dft  == 0 || fwrite(cur.data_dft.data(),  1, len_dft,  f) == len_dft);
        ok = ok && (len_spec == 0 || fwrite(cur.data_spec.data(), 1, len_spec, f) == len_spec);

        if (!ok) {
            fclose(f);
            std::filesystem::remove(tmp, ec);
            return;
        }

        off += CSPILL_HDR_SIZE + len_tgt + len_dft + len_spec;
    }

    const bool ok = fclose(f) == 0;

    if (!ok) {
        std::filesystem::remove(tmp, ec);
        return;
    }

    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return;
    }

    // keep every recorded offset valid: rewrite moves the live segments to the
    // start of the file; the stride matches the write loop above
    int64_t off_new = 0;
    for (auto & cur : prompt.checkpoints) {
        if (!cur.spilled()) {
            continue;
        }
        cur.spill_off = off_new;
        off_new += (int64_t) CSPILL_HDR_SIZE + (int64_t) cur.data_tgt.size() + (int64_t) cur.data_dft.size() + (int64_t) cur.data_spec.size();
    }
}

bool server_prompt_checkpoints_rehydrate(server_prompt & prompt) {
    bool ok = true;

    for (auto & cur : prompt.checkpoints) {
        if (cur.spilled()) {
            ok = server_ckpt_spill_load(cur, prompt.checkpoint_spill_path) && ok;
        }
    }

    return ok;
}

void server_prompt_checkpoints_wipe(server_prompt & prompt) {
    for (auto & cur : prompt.checkpoints) {
        cur.spill_off = -1;
        cur.spill_len = 0;
    }
    prompt.checkpoint_spill_path.clear();
}
