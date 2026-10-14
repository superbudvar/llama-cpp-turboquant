#include "server-task.h"
#include "server-common.h"
#include "server-prompt-cache-store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

// disk prompt cache
//
// each cache entry is stored as <sha256-of-text-tokens>.dpc in the cache directory.
// only text-only prompts (no media chunks) are persisted - media placeholder tokens
// cannot be reconstructed from token ids alone. entry numbers are streamed with
// FILE* writes so large KV blobs are not copied into an extra buffer.
//


// format constants live in server-task.h (writer, reader and tests share them)

// implement the checkpoint section codec below the format constants; both the
// .dpc full-state files (v3) and the .dts store sidecars (v2) use it so the
// layouts cannot drift from each other or from the tests.
bool disk_write_ckpt_section(FILE * f, const std::list<common_prompt_checkpoint> & checkpoints, uint32_t n_ckpt) {
    bool ok = true;

    uint32_t i = 0;
    for (const auto & ckpt : checkpoints) {
        if (i++ >= n_ckpt) {
            break;
        }
        const int64_t  ck_tokens = ckpt.n_tokens;
        const int32_t  id_task   = ckpt.id_task;
        const int32_t  pos_min   = ckpt.pos_min;
        const int32_t  pos_max   = ckpt.pos_max;
        const uint64_t len_tgt   = ckpt.data_tgt.size();
        const uint64_t len_dft   = ckpt.data_dft.size();
        const uint64_t len_spec  = ckpt.data_spec.size();

        ok = ok && fwrite(&ck_tokens, 8, 1, f) == 1;
        ok = ok && fwrite(&id_task,   4, 1, f) == 1;
        ok = ok && fwrite(&pos_min,   4, 1, f) == 1;
        ok = ok && fwrite(&pos_max,   4, 1, f) == 1;
        ok = ok && fwrite(&len_tgt,   8, 1, f) == 1;
        ok = ok && fwrite(&len_dft,   8, 1, f) == 1;
        ok = ok && fwrite(&len_spec,  8, 1, f) == 1;
        ok = ok && (len_tgt == 0 || fwrite(ckpt.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
        ok = ok && (len_dft == 0 || fwrite(ckpt.data_dft.data(),  1, len_dft,  f) == len_dft);
        ok = ok && (len_spec == 0 || fwrite(ckpt.data_spec.data(), 1, len_spec, f) == len_spec);
    }

    return ok;
}

// number of REGION_CKPT chunks a checkpoint's payloads split into: each stream
// is tiled at MAX_REC_LEN, and an empty stream contributes nothing. the writer
// (to size ids.reserve) and the restore (to validate the trailing chunk count)
// must agree on this exact accounting.
uint64_t disk_ckpt_chunk_count(uint32_t n_ckpt, const std::list<common_prompt_checkpoint> & checkpoints) {
    uint64_t n = 0;

    uint32_t i = 0;
    for (const auto & ckpt : checkpoints) {
        if (i++ >= n_ckpt) {
            break;
        }
        n += (ckpt.data_tgt.size()  + (prompt_cache_chunk_store::MAX_REC_LEN - 1)) / prompt_cache_chunk_store::MAX_REC_LEN;
        n += (ckpt.data_dft.size()  + (prompt_cache_chunk_store::MAX_REC_LEN - 1)) / prompt_cache_chunk_store::MAX_REC_LEN;
        n += (ckpt.data_spec.size() + (prompt_cache_chunk_store::MAX_REC_LEN - 1)) / prompt_cache_chunk_store::MAX_REC_LEN;
    }

    return n;
}

// shared .dts v4 header parse: reads magic(4)+version(4)+n_tokens(4)+len_main(8)+
// len_drft(8)+n_ckpt(4)+last_access(8) from the current file position.
// both load_from_disk_tier and disk_read_token_index must use this — they cannot
// drift because they call the same function.
bool disk_parse_dts_header(FILE * f, dts_header & h) {
    h = dts_header{};
    h.ok = fread(&h.magic,    4, 1, f) == 1 &&
           fread(&h.version,  4, 1, f) == 1 &&
           fread(&h.n_tokens, 4, 1, f) == 1 &&
           fread(&h.len_main, 8, 1, f) == 1 &&
           fread(&h.len_drft, 8, 1, f) == 1 &&
           fread(&h.n_ckpt,  4, 1, f) == 1 &&
           h.magic == DTS_MAGIC && h.version == DTS_VERSION &&
           h.n_tokens <= DPC_MAX_TOKENS && h.n_ckpt <= DPC_MAX_CHECKPOINTS;

    // v4: last_access_unix (8 bytes) — must be consumed before token data
    if (h.ok) {
        h.ok = fread(&h.last_access, 8, 1, f) == 1;
    }
    return h.ok;
}

// .dts v3 checkpoint section: metadata headers only, no payload bytes. the
// payloads live as REGION_CKPT chunks in the store; a token sidecar that never
// exceeds ~46 KiB per 1024 checkpoints cannot meaningfully threaten disk-full.
bool disk_write_ckpt_meta(FILE * f, const std::list<common_prompt_checkpoint> & checkpoints, uint32_t n_ckpt) {
    bool ok = true;

    uint32_t i = 0;
    for (const auto & ckpt : checkpoints) {
        if (i++ >= n_ckpt) {
            break;
        }
        const int64_t  ck_tokens = ckpt.n_tokens;
        const int32_t  id_task   = ckpt.id_task;
        const int32_t  pos_min   = ckpt.pos_min;
        const int32_t  pos_max   = ckpt.pos_max;
        const uint64_t len_tgt   = ckpt.data_tgt.size();
        const uint64_t len_dft   = ckpt.data_dft.size();
        const uint64_t len_spec  = ckpt.data_spec.size();

        ok = ok && fwrite(&ck_tokens, 8, 1, f) == 1;
        ok = ok && fwrite(&id_task,   4, 1, f) == 1;
        ok = ok && fwrite(&pos_min,   4, 1, f) == 1;
        ok = ok && fwrite(&pos_max,   4, 1, f) == 1;
        ok = ok && fwrite(&len_tgt,   8, 1, f) == 1;
        ok = ok && fwrite(&len_dft,   8, 1, f) == 1;
        ok = ok && fwrite(&len_spec,  8, 1, f) == 1;
    }

    return ok;
}

bool disk_read_ckpt_meta(FILE * f, uint32_t n_ckpt, std::list<common_prompt_checkpoint> & checkpoints) {
    auto rd = [&](void * dst, size_t n) { return fread(dst, 1, n, f) == n; };

    bool ok = n_ckpt <= DPC_MAX_CHECKPOINTS;

    // sized to the header lengths so the restore knows how many REGION_CKPT
    // chunks each stream reassembles and can validate the reassembly byte count
    std::list<common_prompt_checkpoint> parsed;

    for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
        auto & ckpt = parsed.emplace_back();

        uint64_t len_tgt  = 0;
        uint64_t len_dft  = 0;
        uint64_t len_spec = 0;

        ok = ok && rd(&ckpt.n_tokens, 8) && rd(&ckpt.id_task, 4) &&
              rd(&ckpt.pos_min, 4) && rd(&ckpt.pos_max, 4) &&
              rd(&len_tgt, 8) && rd(&len_dft, 8) && rd(&len_spec, 8);

        ok = ok && len_tgt <= (1ull << 34) && len_dft <= (1ull << 34) && len_spec <= (1ull << 34);

        if (ok) {
            ckpt.data_tgt.resize(len_tgt);
            ckpt.data_dft.resize(len_dft);
            ckpt.data_spec.resize(len_spec);
        }
    }

    if (ok) {
        checkpoints.splice(checkpoints.end(), parsed);
    }

    return ok;
}

uint64_t disk_ckpt_meta_size(uint32_t n_ckpt) {
    return ((uint64_t) n_ckpt) * 44;
}

bool disk_read_ckpt_section(FILE * f, uint32_t n_ckpt, std::list<common_prompt_checkpoint> & checkpoints) {
    auto rd = [&](void * dst, size_t n) { return fread(dst, 1, n, f) == n; };

    bool ok = n_ckpt <= DPC_MAX_CHECKPOINTS;

    std::list<common_prompt_checkpoint> parsed;

    for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
        auto & ckpt = parsed.emplace_back();

        uint64_t len_tgt  = 0;
        uint64_t len_dft  = 0;
        uint64_t len_spec = 0;

        ok = ok && rd(&ckpt.n_tokens, 8) && rd(&ckpt.id_task, 4) &&
              rd(&ckpt.pos_min, 4) && rd(&ckpt.pos_max, 4) &&
              rd(&len_tgt, 8) && rd(&len_dft, 8) && rd(&len_spec, 8);

        ok = ok && len_tgt <= (1ull << 34) && len_dft <= (1ull << 34) && len_spec <= (1ull << 34);

        if (ok) {
            ckpt.data_tgt.resize(len_tgt);
            ckpt.data_dft.resize(len_dft);
            ckpt.data_spec.resize(len_spec);
            ok = ok && (len_tgt == 0  || fread(ckpt.data_tgt.data(),  1, len_tgt,  f) == len_tgt);
            ok = ok && (len_dft == 0  || fread(ckpt.data_dft.data(),  1, len_dft,  f) == len_dft);
            ok = ok && (len_spec == 0 || fread(ckpt.data_spec.data(), 1, len_spec, f) == len_spec);
        }
    }

    if (ok) {
        checkpoints.splice(checkpoints.end(), parsed);
    }

    return ok;
}

uint64_t disk_ckpt_section_size(const std::list<common_prompt_checkpoint> & checkpoints, uint32_t n_ckpt) {
    uint64_t sz = 0;

    uint32_t i = 0;
    for (const auto & ckpt : checkpoints) {
        if (i++ >= n_ckpt) {
            break;
        }
        sz += 44 + ckpt.data_tgt.size() + ckpt.data_dft.size() + ckpt.data_spec.size();
    }

    return sz;
}

// compact SHA-256 (FIPS 180-4), single-shot. mirrors the core in common/fit-cache.cpp
uint32_t dpc_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

namespace {

uint32_t dpc_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void dpc_block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t) p[i * 4 + 0] << 24) |
               ((uint32_t) p[i * 4 + 1] << 16) |
               ((uint32_t) p[i * 4 + 2] <<  8) |
               ((uint32_t) p[i * 4 + 3] <<  0);
    }
    for (size_t i = 16; i < 64; i++) {
        const uint32_t s0 = dpc_rotr(w[i - 15], 7) ^ dpc_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = dpc_rotr(w[i - 2], 17)  ^ dpc_rotr(w[i - 2], 19)   ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    for (size_t i = 0; i < 64; i++) {
        const uint32_t S1 = dpc_rotr(e, 6) ^ dpc_rotr(e, 11) ^ dpc_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + S1 + ch + dpc_sha256_k[i] + w[i];
        const uint32_t S0 = dpc_rotr(a, 2) ^ dpc_rotr(a, 13) ^ dpc_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

std::string dpc_sha256_hex(const uint8_t * data, size_t len) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    const uint64_t bitlen = (uint64_t) len * 8;
    std::vector<uint8_t> buf(data, data + len);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0x00);
    }
    for (int i = 7; i >= 0; i--) {
        buf.push_back((uint8_t) (bitlen >> (i * 8)));
    }

    for (size_t off = 0; off < buf.size(); off += 64) {
        dpc_block(h, buf.data() + off);
    }

    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (size_t i = 0; i < 8; i++) {
        const uint32_t v = h[i];
        const uint8_t b[4] = {
            (uint8_t) (v >> 24), (uint8_t) (v >> 16), (uint8_t) (v >> 8), (uint8_t) (v >> 0),
        };
        for (int j = 0; j < 4; j++) {
            out.push_back(hex[b[j] >> 4]);
            out.push_back(hex[b[j] & 0x0f]);
        }
    }
    return out;
}

// raw sha256 (32 bytes out); used for the chunk-store lineage chain
void dpc_sha256_raw(const uint8_t * data, size_t len, uint8_t * out) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    const uint64_t bitlen = (uint64_t) len * 8;
    std::vector<uint8_t> buf(data, data + len);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0x00);
    }
    for (int i = 7; i >= 0; i--) {
        buf.push_back((uint8_t) (bitlen >> (i * 8)));
    }

    for (size_t off = 0; off < buf.size(); off += 64) {
        dpc_block(h, buf.data() + off);
    }

    for (size_t i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t) (h[i] >> 24);
        out[i * 4 + 1] = (uint8_t) (h[i] >> 16);
        out[i * 4 + 2] = (uint8_t) (h[i] >>  8);
        out[i * 4 + 3] = (uint8_t) (h[i] >>  0);
    }
}

} // namespace

void server_prompt_cache_sha256_raw(const uint8_t * data, size_t len, uint8_t * out) {
    dpc_sha256_raw(data, len, out);
}

// slice a llama_state_seq blob into fixed 128-token cell sheets, in canonical
// (blob-order) slice order. the slices tile [0, blob_size) contiguously, so
// reassembly by concatenation is byte-exact for any accepted blob. acceptance
// requires the single-stream write-through layout the store persists; anything
// else - a draft stream, a non-consecutive or multi-sequence meta section, a
// transposed V, an out-of-range header, or an ambiguous meta stride - returns
// false and the caller keeps the legacy .dpc writer.
bool slice_state_blob(const uint8_t * blob, size_t blob_size, size_t n_tokens,
                      std::vector<state_blob_slice> & out, uint8_t region_shift) {
    constexpr uint32_t SEQ_MAGIC = 0xaf143cd8;
    constexpr uint64_t MAX_ROW   = 1ull << 24;
    constexpr uint32_t CELL      = 128;
    out.clear();

    if (blob_size < 16) {
        return false;
    }
    if (cc_io::get_u32(blob) != SEQ_MAGIC ||
        cc_io::get_u32(blob + 8) != 1 ||
        cc_io::get_u32(blob + 12) != (uint32_t) n_tokens) {
        return false;
    }

    // real llama_state_seq layout: magic(4) + seq_id(4) + n_stream(4) +
    // cell_count(4) = 16 bytes prefix; meta rows begin immediately after
    const size_t meta_start = 16;

    // meta stride: 12 bytes/cell without M-RoPE ext, 20 with ext. the right
    // stride makes every cell a single-sequence entry (n_seq_id == 1) with
    // pos == its row index, which is exactly the contiguous write-through
    // layout the store requires. acceptance is exclusive: two candidates that
    // validate are a layout the walk cannot prove, none is unsupported
    size_t meta_stride = 0;
    size_t d           = 0;
    const size_t strides[] = { 12, 20 };
    for (const size_t cand : strides) {
        size_t off = meta_start;
        bool   ok  = true;
        for (size_t i = 0; i < n_tokens && ok; ++i) {
            if (off + cand > blob_size) {
                ok = false;
                break;
            }
            // a write-through cell has pos == i and n_seq_id == 1
            if (cc_io::get_u32(blob + off) != (uint32_t) i ||
                cc_io::get_u32(blob + off + 4) != 1) {
                ok = false;
                break;
            }
            off += cand;
        }
        if (ok) {
            if (meta_stride != 0) {
                return false;
            }
            meta_stride = cand;
            d           = off;
        }
    }
    if (meta_stride == 0) {
        return false;
    }

    if (d + 8 > blob_size) {
        return false;
    }
    const uint32_t v_trans = cc_io::get_u32(blob + d);
    const uint32_t n_layer = cc_io::get_u32(blob + d + 4);
    if (v_trans != 0 || n_layer == 0 || n_layer > 4096) {
        return false;
    }

    // K blocks: one 12 B header then n_tokens rows of row_size bytes, per layer.
    // the header must be a plausible ggml type and the data block must fit
    size_t k_end = d + 8;
    std::vector<uint64_t> k_off;
    std::vector<uint64_t> k_row;
    k_off.reserve(n_layer);
    k_row.reserve(n_layer);
    for (uint32_t l = 0; l < n_layer; ++l) {
        if (k_end + 12 > blob_size) {
            return false;
        }
        const int32_t type = (int32_t) cc_io::get_u32(blob + k_end);
        const uint64_t row = cc_io::get_u64(blob + k_end + 4);
        if (type < 0 || type >= (int32_t) GGML_TYPE_COUNT || row == 0 || row > MAX_ROW ||
            (uint64_t) n_tokens * row > blob_size - k_end - 12) {
            return false;
        }
        k_off.push_back(k_end);
        k_row.push_back(row);
        k_end += 12 + (uint64_t) n_tokens * row;
    }

    // V blocks: present for non-MLA models, absent for MLA. parse them
    // best-effort with the same header checks; any failure turns the whole
    // remainder into one opaque tail chunk, which reassembles equally
    // byte-exactly on restore
    std::vector<uint64_t> v_off;
    std::vector<uint64_t> v_row;
    size_t v_end = k_end;
    for (uint32_t l = 0; l < n_layer; ++l) {
        if (v_end + 12 > blob_size) {
            v_off.clear();
            v_row.clear();
            break;
        }
        const int32_t type = (int32_t) cc_io::get_u32(blob + v_end);
        const uint64_t row = cc_io::get_u64(blob + v_end + 4);
        if (type < 0 || type >= (int32_t) GGML_TYPE_COUNT || row == 0 || row > MAX_ROW ||
            (uint64_t) n_tokens * row > blob_size - v_end - 12) {
            v_off.clear();
            v_row.clear();
            break;
        }
        v_off.push_back(v_end);
        v_row.push_back(row);
        v_end += 12 + (uint64_t) n_tokens * row;
    }
    const size_t attn_end = v_off.size() == n_layer ? v_end : k_end;

    const uint32_t n_cells   = (uint32_t) ((n_tokens + CELL - 1) / CELL);
    const uint32_t n_tok32   = (uint32_t) n_tokens;

    // every emitted region is offset by region_shift. the main store uses 0;
    // the draft (MTP) stream uses REGION_DRAFT so its K/V/meta/header/tail
    // sheets live in a disjoint chunk-key space from the main stream even
    // though the walker sees an identical layout
    auto sh = [region_shift](prompt_cache_chunk_store::region r) {
        return (prompt_cache_chunk_store::region) ((uint8_t) r + region_shift);
    };

    // 1) fixed prefix [0, meta_start): magic, seq_id, n_stream, cell_count
    out.push_back({ sh(prompt_cache_chunk_store::REGION_HEADER), 0, 0, n_tok32, 0, (uint64_t) meta_start });

    // 2) meta rows [meta_start, d): one sheet per 128-token cell
    for (uint32_t c = 0; c < n_cells; ++c) {
        const uint32_t lo = c * CELL;
        const uint32_t hi = std::min(n_tok32, lo + CELL);
        out.push_back({ sh(prompt_cache_chunk_store::REGION_META_ROW), 0, lo, hi,
                        (uint64_t) (meta_start + (size_t) lo * meta_stride),
                        (uint64_t) ((size_t) (hi - lo) * meta_stride) });
    }

    // 3) v_trans + n_layer table [d, d + 8); keyed apart from the prefix so the
    //    two header chunks never share a dedup identity
    out.push_back({ sh(prompt_cache_chunk_store::REGION_HEADER), 0, 0, 0, (uint64_t) d, 8 });

    // 4) + 5) K and V cell sheets: the 12 B block header folds into cell 0 so
    //    the sheet ranges stay contiguous byte ranges of the blob
    auto emit_cells = [&](prompt_cache_chunk_store::region region, uint16_t layer, uint64_t block_off, uint64_t row) {
        const uint64_t cell_bytes = (uint64_t) CELL * row;
        for (uint32_t c = 0; c < n_cells; ++c) {
            const uint32_t lo = c * CELL;
            const uint32_t hi = std::min(n_tok32, lo + CELL);
            const uint64_t n  = (uint64_t) (hi - lo);
            out.push_back({ sh(region), layer, lo, hi,
                            block_off + (c == 0 ? 0 : 12) + (uint64_t) c * cell_bytes,
                            n * row + (c == 0 ? 12 : 0) });
        }
    };
    for (uint32_t l = 0; l < n_layer; ++l) {
        emit_cells(prompt_cache_chunk_store::REGION_K, (uint16_t) l, k_off[l], k_row[l]);
    }
    for (uint32_t l = 0; l < v_off.size(); ++l) {
        emit_cells(prompt_cache_chunk_store::REGION_V, (uint16_t) l, v_off[l], v_row[l]);
    }

    // 6) recurrent-state tail [attn_end, blob_size); absent when the model has
    //    no recurrent head
    if (attn_end < blob_size) {
        out.push_back({ sh(prompt_cache_chunk_store::REGION_TAIL), 0, 0, n_tok32,
                        (uint64_t) attn_end, (uint64_t) (blob_size - attn_end) });
    }
    return true;
}

// n_*(): not used by the store contract itself but kept for parity with the
// legacy per-state index that callers build from get_stats()

static void disk_write_meta_file(const std::string & dir, const std::string & fingerprint);

server_prompt_cache_disk::server_prompt_cache_disk(size_t disk_max_mib, size_t disk_store_mib) {
    disk_max_size = 1024ull * 1024ull * disk_max_mib;
    disk_max_store_size = disk_store_mib > 0 ? 1024ull * 1024ull * disk_store_mib : (disk_max_size * 6) / 10;
    start_vacuum_thread();
}

server_prompt_cache_disk::~server_prompt_cache_disk() {
    drain_disk_writes();
    if (vacuum_thread_.joinable()) {
        vacuum_stop_.store(true, std::memory_order_relaxed);
        vacuum_cv_.notify_one();
        vacuum_thread_.join();
    }
}

bool server_prompt_cache_disk::save_to_disk(const std::string & dir, const std::string & fingerprint) const {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        fs::create_directories(dir, ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: failed to create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
            return false;
        }
    }

    // v4 only: all data is already in the chunk store (cache.dcp + .dts sidecars).
    // this call only updates the meta fingerprint so the directory is valid on boot.
    disk_write_meta_file(dir, fingerprint);
    return true;
}

void server_prompt_cache_disk::prune_disk_budget(const std::string & dir, uint64_t min_free) const {
    namespace fs = std::filesystem;

    // the pool root is the parent of the per-preset dir (cache_disk_path/<name>)
    const fs::path root = fs::path(dir).parent_path();
    prune_pool(root.string(), dir, true, nullptr, min_free);
}

void server_prompt_cache_disk::disk_prune_dpc(const std::string & dir, std::vector<std::array<uint8_t, 32>> * evicted, uint64_t min_free) const {
    namespace fs = std::filesystem;

    const fs::path root = fs::path(dir).parent_path();
    prune_pool(root.string(), dir, false, evicted, min_free);
}

void server_prompt_cache_disk::prune_pool(const std::string & root, const std::string & own_dir, bool include_own_ccn,
                                    std::vector<std::array<uint8_t, 32>> * evicted, uint64_t min_free) const {
    namespace fs = std::filesystem;

    if (disk_max_size == 0) {
        // 0 is documented as "no limit" (common.h); on a shared root that silently
        // unenforces the pool for every preset. warn once instead of asserting,
        // asserts are compiled out in Release
        if (!disk_no_limit_warned_ && fs::is_directory(root)) {
            std::error_code ec_w;
            for (const auto & sd : fs::directory_iterator(root, ec_w)) {
                if (sd.is_directory(ec_w)) {
                    disk_no_limit_warned_ = true;
                    SRV_WRN(" - disk prompt cache: disk_max_size == 0 (no limit) on a shared root '%s' - pool budget is not enforced\n", root.c_str());
                    break;
                }
            }
        }
        // the budget pass below is disabled, but the free-space pass still runs
    }

    struct pool_file {
        fs::path path;
        fs::file_time_type mtime;
        uint64_t size;
        bool is_ccn;
    };

    std::error_code ec;
    std::vector<pool_file> files;
    uint64_t used = 0;

    const fs::path own_path(own_dir);
    // knob B: the own store footprint (.dcp + .dci) and its own .dts sidecars
    uint64_t own_store = 0;
    std::vector<pool_file> own_dts;

    for (const auto & sd : fs::directory_iterator(root, ec)) {
        if (!sd.is_directory(ec)) {
            continue;
        }
        const bool is_own_dir = sd.path() == own_path;

        for (const auto & entry : fs::directory_iterator(sd.path(), ec)) {
            const auto ext = entry.path().extension();
            if (ext == ".dpc" || ext == ".dts") {
                const auto size = fs::file_size(entry.path(), ec);
                if (ec) {
                    continue;
                }
                used += (uint64_t) size;
                files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size, false });
                if (is_own_dir && ext == ".dts") {
                    own_dts.push_back(files.back());
                }
            } else if (ext == ".ccn" && is_own_dir && include_own_ccn) {
                const auto size = fs::file_size(entry.path(), ec);
                if (ec) {
                    continue;
                }
                used += (uint64_t) size;
                files.push_back({ entry.path(), fs::last_write_time(entry.path(), ec), (uint64_t) size, true });
            } else if (ext == ".dcp" || ext == ".dci") {
                // chunk store files count toward the budget but are never evicted
                const auto size = fs::file_size(entry.path(), ec);
                if (!ec) {
                    used += (uint64_t) size;
                    if (is_own_dir) {
                        own_store += (uint64_t) size;
                    }
                }
            }
        }
    }

    // evict oldest first (shared with the free-space pass below)
    const auto eval_oldest = [&files]() {
        std::sort(files.begin(), files.end(), [](const pool_file & a, const pool_file & b) {
            return a.mtime < b.mtime;
        });
    };

    // remove one file, returning its size if the removal freed the bytes.
    // .dts fingerprint extraction must happen before the file is gone so the
    // caller can drop the stale state from the chunk store
    const auto evict_one = [this, evicted](pool_file & file) -> uint64_t {
        if (evicted && file.path.extension() == ".dts") {
            FILE * f = fopen(file.path.string().c_str(), "rb");
            if (f) {
                uint32_t magic = 0, version = 0, n_tokens = 0;
                if (fread(&magic, 4, 1, f) == 1 && magic == DTS_MAGIC &&
                    fread(&version, 4, 1, f) == 1 && version == DTS_VERSION &&
                    fread(&n_tokens, 4, 1, f) == 1 && n_tokens > 0 && n_tokens <= 131072) {
                        // skip len_main(8) + len_drft(8) + n_ckpt(4) + last_access(8) = 28 bytes
                        fseek(f, 28, SEEK_CUR);
                        std::vector<int32_t> tokens(n_tokens);
                        if (fread(tokens.data(), sizeof(int32_t), n_tokens, f) == n_tokens) {
                            std::array<uint8_t, 32> fp;
                            dpc_sha256_raw((const uint8_t *) tokens.data(), n_tokens * sizeof(int32_t), fp.data());
                            evicted->push_back(fp);
                        }
                    }
                fclose(f);
            }
        }

        std::error_code ec_remove;
        fs::remove(file.path, ec_remove);
        if (ec_remove) {
            return 0; // file still occupied budget, keep its size
        }
        return file.size;
    };

    eval_oldest();

    // knob B (store cap): a .dcp store may own at most disk_max_store_size
    // bytes; when it exceeds that, evict its own .dts sidecars first (they are
    // the store's debt - dropping them lets the caller compact the .dcp) before
    // the shared-pool pass below evicts foreign entries. the footprint is the
    // .dcp+.dci plus the remaining own .dts, so the pass converges as the caller
    // compacts after drop_evicted. only runs when the caller can consume the
    // evicted fingerprints (else the states would leak in the store).
    if (disk_max_store_size > 0 && evicted != nullptr && !own_dts.empty()) {
        uint64_t own_foot = own_store;
        for (const auto & dt : own_dts) {
            own_foot += dt.size;
        }
        if (own_foot > disk_max_store_size) {
            std::sort(own_dts.begin(), own_dts.end(), [](const pool_file & a, const pool_file & b) {
                return a.mtime < b.mtime;
            });
            for (auto & dt : own_dts) {
                if (own_foot <= disk_max_store_size) {
                    break;
                }
                const uint64_t freed = evict_one(dt);
                if (freed > 0) {
                    own_foot -= freed;
                    used     -= freed;
                }
            }
            SRV_INF(" - disk prompt cache: store '%s' over cap (%.3f MiB > %.3f MiB), evicted %zu own .dts sidecars\n",
                    own_dir.c_str(), own_foot / (1024.0 * 1024.0), disk_max_store_size / (1024.0 * 1024.0),
                    std::count_if(own_dts.begin(), own_dts.end(), [&](const pool_file & f) { return !fs::exists(f.path); }));
        }
    }

    if (disk_max_size > 0 && used > disk_max_size) {
        for (auto & file : files) {
            if (used <= disk_max_size) {
                break;
            }
            used -= evict_one(file);
        }

        SRV_INF(" - disk prompt cache: pruned shared pool '%s' to %.3f MiB (budget %.3f MiB)\n",
                root.c_str(), used / (1024.0 * 1024.0), disk_max_size / (1024.0 * 1024.0));
    }

    // free-space pass: even under budget, a full volume must evict the oldest
    // .dpc/.dts entries until the requested bytes are available. .ccn files are
    // only released by the budget pass (owned by the main thread); .dcp/.dci
    // are never evicted here (store compaction owns them)
    if (min_free > 0) {
        std::error_code ec_free;
        auto available = fs::space(root, ec_free).available;
        if (!ec_free && available < min_free) {
            for (auto & file : files) {
                if (available >= min_free) {
                    break;
                }
                if (file.is_ccn) {
                    continue;
                }
                evict_one(file);
                available = fs::space(root, ec_free).available;
                if (ec_free) {
                    break;
                }
            }
            SRV_INF(" - disk prompt cache: evicted shared pool '%s' for free space (now %.3f MiB free, need %.3f MiB)\n",
                    root.c_str(), (double) available / (1024.0 * 1024.0), (double) min_free / (1024.0 * 1024.0));
        }
    }
}

server_prompt_cache_save_result server_prompt_cache_disk::commit_disk_state(server_prompt_cache_state * state) {
    // only the pending write-through entry gets committed here; RAM-tier entries
    // (and callers with no pending scratch) are passed through untouched
    if (state != disk_state_.get()) {
        return server_prompt_cache_save_result::stored_in_ram;
    }

    // media placeholders cannot be rebuilt from token ids, skip such entries
    if (state->prompt.tokens.has_media()) {
        SRV_TRC("%s", " - disk prompt cache: skipping media prompt\n");
        disk_state_.reset();
        return server_prompt_cache_save_result::not_stored;
    }

    if (disk_dir_.empty()) {
        disk_state_.reset();
        return server_prompt_cache_save_result::not_stored;
    }

    // move state into the job and free RAM before the write is done
    disk_write_job job;
    job.kind        = disk_write_job::type::state;
    job.state       = std::move(disk_state_);
    job.dir         = disk_dir_;
    job.fingerprint = disk_fingerprint_;

    disk_state_.reset();

    start_disk_thread();

    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_jobs_.push_back(std::move(job));
    }
    disk_jobs_cv_.notify_one();

    // Queue admission is not a durable save. Keep the live slot until a later
    // prompt-cache match proves that its state can be discarded.
    return server_prompt_cache_save_result::queued_to_disk;
}

void server_prompt_cache_disk::start_disk_thread() {
    if (disk_thread_.joinable()) {
        return;
    }
    disk_thread_stop_ = false;
    try {
        disk_thread_ = std::thread([this] { disk_thread_main(); });
    } catch (const std::system_error & e) {
        SRV_WRN(" - disk prompt cache: failed to start writer thread: %s\n", e.what());
    }
}

void server_prompt_cache_disk::drain_disk_writes() {
    if (!disk_thread_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_thread_stop_ = true;
    }
    disk_jobs_cv_.notify_one();
    disk_thread_.join();
}

void server_prompt_cache_disk::start_vacuum_thread() {
    if (disk_max_size == 0) {
        return;   // no disk budget: vacuum disabled
    }
    if (vacuum_thread_.joinable()) {
        return;
    }
    try {
        vacuum_thread_ = std::thread([this] {
            constexpr auto INTERVAL = std::chrono::minutes(5);
            std::unique_lock<std::mutex> lk(vacuum_mtx_);
            while (!vacuum_stop_.load(std::memory_order_relaxed)) {
                vacuum_cv_.wait_for(lk, INTERVAL, [this] {
                    return vacuum_stop_.load(std::memory_order_relaxed);
                });
                if (vacuum_stop_.load(std::memory_order_relaxed)) {
                    break;
                }
                lk.unlock();
                vacuum_sweep_all();
                lk.lock();
            }
        });
    } catch (const std::system_error & e) {
        SRV_WRN(" - disk prompt cache: failed to start vacuum thread: %s\n", e.what());
    }
}

void server_prompt_cache_disk::vacuum_sweep_all() {
    namespace fs = std::filesystem;
    if (disk_dir_.empty()) {
        return;
    }
    const fs::path root = fs::path(disk_dir_).parent_path();
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return;
    }
    for (const auto & sd : fs::directory_iterator(root, ec)) {
        if (!sd.is_directory(ec)) {
            continue;
        }
        const auto sub = sd.path();
        // each subdir gets its own temporary store so we never conflict with
        // the writer thread's disk_store_
        prompt_cache_chunk_store vacuum_store;
        if (!vacuum_store.open(sub.string(), 0)) {
            continue;  // no .dcp here or locked by writer
        }
        if (vacuum_store.should_compact(disk_max_size)) {
            vacuum_store.compact();
        }
        vacuum_store.close();
    }
}

bool server_prompt_cache_disk::enqueue_sidecar(const std::string & path, std::vector<uint8_t> bytes) {
    start_disk_thread();
    if (!disk_thread_.joinable()) {
        return false;
    }
    disk_write_job job;
    job.kind  = disk_write_job::type::sidecar;
    job.path  = path;
    job.bytes = std::move(bytes);
    {
        std::lock_guard<std::mutex> lock(disk_jobs_mtx_);
        disk_jobs_.push_back(std::move(job));
    }
    disk_jobs_cv_.notify_one();
    return true;
}

// base lineage for the chunk store: the low 8 bytes of sha256 of the prompt
// cache fingerprint. a config change changes the fingerprint, hence the
// lineage, which resets the store exactly like the .dpc/.ccn wipe
static uint64_t fingerprint_lineage(const std::string & fingerprint) {
    uint8_t h[32];
    dpc_sha256_raw((const uint8_t *) fingerprint.data(), fingerprint.size(), h);
    return cc_io::get_u64(h);
}

// keep the meta fingerprint current so later saves do not treat the directory
// as a config mismatch; shared by the legacy and chunk-store writers
static void disk_write_meta_file(const std::string & dir, const std::string & fingerprint) {
    namespace fs = std::filesystem;
    const std::string meta_path = (fs::path(dir) / "meta").string();
    const std::string meta_tmp  = meta_path + ".tmp";

    std::ofstream out(meta_tmp, std::ios::binary);
    out.write(fingerprint.data(), (std::streamsize) fingerprint.size());
    out.close();

    std::error_code ec;
    fs::rename(meta_tmp, meta_path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: failed to update meta: %s\n", ec.message().c_str());
    }
}

// write the <fp>.dts store index for a chunk-store state. called after the
// store batch commits, so a sidecar present always implies a durable state; a
// failed sidecar only costs the entry at restore time (state stays on disk).
// v3 (current DTS_VERSION): only the 44-byte checkpoint metadata headers ride
// in the sidecar - the payloads are REGION_CKPT chunks in the store, so the
// sidecar stays tiny even for a full checkpoint set and the store batch is the
// single source of dedup. a restored v3 state reassembles the payloads from
// the trailing chunk ids.
bool disk_write_token_index(const std::string & dir, const std::string & fp_hex,
                                   const llama_tokens & tokens, uint64_t len_main, uint64_t len_drft,
                                   const std::list<common_prompt_checkpoint> & checkpoints,
                                   uint64_t last_access) {
    namespace fs = std::filesystem;
    const std::string dts_path = (fs::path(dir) / (fp_hex + ".dts")).string();
    const std::string dts_tmp  = dts_path + ".tmp";

    FILE * f = fopen(dts_tmp.c_str(), "wb");
    if (f == nullptr) {
        SRV_WRN(" - disk prompt cache: cannot create token index '%s': %s\n", dts_tmp.c_str(), strerror(errno));
        return false;
    }

    const uint32_t n_tokens = (uint32_t) tokens.size();
    uint32_t n_ckpt = (uint32_t) std::min<size_t>(checkpoints.size(), DPC_MAX_CHECKPOINTS);

    // v4: always write last_access + checkpoint metadata (44-byte headers, no payloads)
    bool ok = fwrite(&DTS_MAGIC,   4, 1, f) == 1;
    ok = ok && fwrite(&DTS_VERSION, 4, 1, f) == 1;
    ok = ok && fwrite(&n_tokens,    4, 1, f) == 1;
    ok = ok && fwrite(&len_main,    8, 1, f) == 1;
    ok = ok && fwrite(&len_drft,    8, 1, f) == 1;
    ok = ok && fwrite(&n_ckpt,      4, 1, f) == 1;
    ok = ok && fwrite(&last_access,  8, 1, f) == 1;
    ok = ok && (n_tokens == 0 || fwrite(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
    ok = ok && disk_write_ckpt_meta(f, checkpoints, n_ckpt);
    ok = fclose(f) == 0 && ok;

    if (!ok) {
        SRV_WRN(" - disk prompt cache: failed to write token index '%s'\n", dts_path.c_str());
        std::error_code ec;
        fs::remove(dts_tmp, ec);
        return false;
    }

    std::error_code ec;
    fs::rename(dts_tmp, dts_path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: failed to finalize token index '%s': %s\n", dts_path.c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

disk_token_index_result disk_read_token_index(const std::string & path) {
    disk_token_index_result r;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return r;
    }

    if (!disk_parse_dts_header(f, r.hdr)) {
        fclose(f);
        return r;
    }

    if (r.hdr.n_tokens > 0) {
        r.tokens.resize(r.hdr.n_tokens);
        if (fread(r.tokens.data(), sizeof(llama_token), r.hdr.n_tokens, f) != r.hdr.n_tokens) {
            fclose(f);
            return r;
        }
    }

    if (!disk_read_ckpt_meta(f, r.hdr.n_ckpt, r.checkpoints)) {
        fclose(f);
        return r;
    }

    fclose(f);
    return r;
}

bool server_prompt_cache_disk::disk_store_write(disk_write_job & job, const llama_tokens & tokens) {
    namespace fs = std::filesystem;
    auto & state = *job.state;
    if (state.data.main.empty()) {
        return false;
    }

    const uint64_t base = fingerprint_lineage(job.fingerprint);

    // store disabled: stick to the legacy .dpc writer without ever touching
    // cache.dcp/.dci (one-pass rollback) or creating them on disk
    if (!prompt_cache_chunk_store::store_enabled()) {
        return false;
    }

    // open lazily (writer thread first, and later any restore on the reader
    // thread); a config change is a different base lineage and resets the store
    // like the .dpc/.ccn wipe
    {
        std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
        if (!disk_store_) {
            disk_store_ = std::make_unique<prompt_cache_chunk_store>();
        }
        if (!disk_store_->is_open() && !disk_store_->open(job.dir, base)) {
            SRV_WRN(" - disk prompt cache: chunk store unavailable in '%s', using .dpc writer\n", job.dir.c_str());
            return false;
        }
    }
    if (!disk_store_->wants_dedup()) {
        return false;
    }

    std::vector<state_blob_slice> slices;
    if (!slice_state_blob(state.data.main.data(), state.data.main.size(), tokens.size(), slices)) {
        return false;
    }

    // a draft (MTP) stream is sliced with the same walker but its cells are
    // tagged with the REGION_DRAFT offset so its chunks never share a dedup
    // identity with the main stream. the draft blob is also a llama_state_seq
    // blob: its own cell_count sits at byte 12 and its meta rows are
    // consecutive from position 0, so the strict walker gate applies directly
    std::vector<state_blob_slice> drft_slices;
    if (!state.data.drft.empty()) {
        if (state.data.drft.size() < 16) {
            return false;
        }
        const uint32_t drft_cells = cc_io::get_u32(state.data.drft.data() + 12);
        if (drft_cells == 0 || drft_cells > tokens.size() ||
            !slice_state_blob(state.data.drft.data(), state.data.drft.size(), drft_cells, drft_slices,
                              (uint8_t) prompt_cache_chunk_store::REGION_DRAFT)) {
            return false;
        }
    }

    const std::vector<uint64_t> lineages =
        prompt_cache_chunk_store::cell_lineages(base, tokens.data(), tokens.size());
    const uint64_t last_epoch = lineages.empty() ? base : lineages.back();

    // the epoch of a K/V/meta sheet is the token chain up to and including its
    // cell; the header/tail chunks span the whole state and carry the last one.
    // draft regions are the main regions shifted by REGION_DRAFT; normalize
    // before selecting, and note the draft cell grid is the main position grid
    auto epoch_of = [&](const state_blob_slice & sl) -> uint64_t {
        const uint8_t base_region = (uint8_t) ((uint8_t) sl.region >= (uint8_t) prompt_cache_chunk_store::REGION_DRAFT
                                    ? (uint8_t) sl.region - (uint8_t) prompt_cache_chunk_store::REGION_DRAFT
                                    : (uint8_t) sl.region);
        switch (base_region) {
        case prompt_cache_chunk_store::REGION_K:
        case prompt_cache_chunk_store::REGION_V:
        case prompt_cache_chunk_store::REGION_META_ROW:
            return lineages[sl.cell_lo / 128];
        default:
            return last_epoch;
        }
    };

    // one batch per save; abort leaves no trace. disk-full is checked on the
    // appended delta (dedup hits add nothing), then compacted and retried once
    auto write_batch = [&]() -> bool {
        if (!disk_store_->begin_commit()) {
            return false;
        }
        std::vector<uint64_t> ids;
        const uint32_t n_ckpt = (uint32_t) std::min<size_t>(state.prompt.checkpoints.size(), DPC_MAX_CHECKPOINTS);
        ids.reserve(slices.size() + drft_slices.size() + disk_ckpt_chunk_count(n_ckpt, state.prompt.checkpoints));
        auto put_part = [&](const std::vector<state_blob_slice> & part, const uint8_t * blob) -> bool {
            for (const auto & sl : part) {
                prompt_cache_chunk_store::chunk_key key;
                key.region        = sl.region;
                key.layer         = sl.layer;
                key.cell_lo       = sl.cell_lo;
                key.cell_hi       = sl.cell_hi;
                key.lineage_epoch = epoch_of(sl);
                const int64_t id = disk_store_->put_chunk(key, blob + sl.byte_off, sl.byte_len);
                if (id < 0) {
                    return false;
                }
                ids.push_back((uint64_t) id);
            }
            return true;
        };
        if (!put_part(slices, state.data.main.data()) ||
            !put_part(drft_slices, state.data.drft.data())) {
            disk_store_->abort_commit();
            return false;
        }

        // checkpoint payloads dedup across consecutive states: store them as their
        // own chunks (region REGION_CKPT) right after the draft ids. a checkpoint
        // identical between consecutive states shares every id. the .dts v3
        // sidecar then only has to list the metadata. layer packs the stream
        // (0=tgt,1=dft,2=spec) and the sub-chunk ordinal. the key cannot derive
        // lineage from the token chain: checkpoint payloads carry spec/beam/decoder
        // state that is not a pure function of the tokens, so keyed dedup would
        // alias a changed payload at the same span back to the stale chunk. key
        // on a content tag of the tile instead: byte-identical tiles dedup,
        // anything else gets its own chunk
        uint32_t ckpt_i = 0;
        for (const auto & ckpt : state.prompt.checkpoints) {
            if (ckpt_i++ >= n_ckpt) {
                break;
            }

            const uint32_t lo = ckpt.pos_min < 0 ? 0       : (uint32_t) ckpt.pos_min;
            const uint32_t hi = ckpt.pos_max < 0 ? 0       : (uint32_t) ckpt.pos_max;

            auto put_ckpt_stream = [&](uint32_t stream, const std::vector<uint8_t> & payload) -> bool {
                for (uint64_t off = 0; off < payload.size(); off += prompt_cache_chunk_store::MAX_REC_LEN) {
                    prompt_cache_chunk_store::chunk_key key;
                    key.region        = prompt_cache_chunk_store::REGION_CKPT;
                    key.layer         = (uint16_t) (stream << 10 | (uint32_t) (off / prompt_cache_chunk_store::MAX_REC_LEN));
                    key.cell_lo       = lo;
                    key.cell_hi       = hi;
                    const uint64_t len = std::min<uint64_t>(prompt_cache_chunk_store::MAX_REC_LEN, payload.size() - off);
                    uint8_t tag[32];
                    dpc_sha256_raw(payload.data() + off, len, tag);
                    // fold the whole digest so the epoch is a real content tag;
                    // collisions are then a 128-bit-equivalent identity decision
                    key.lineage_epoch = cc_io::get_u64(tag)       ^ cc_io::get_u64(tag + 8) ^
                                        cc_io::get_u64(tag + 16)  ^ cc_io::get_u64(tag + 24);
                    const int64_t id = disk_store_->put_chunk(key, payload.data() + off, len);
                    if (id < 0) {
                        return false;
                    }
                    ids.push_back((uint64_t) id);
                }
                return true;
            };
            if (!put_ckpt_stream(0, ckpt.data_tgt) ||
                !put_ckpt_stream(1, ckpt.data_dft) ||
                !put_ckpt_stream(2, ckpt.data_spec)) {
                disk_store_->abort_commit();
                return false;
            }
        }

        // never commit when the appended bytes would push the volume full
        {
            std::error_code ecs;
            const uint64_t free = fs::space(fs::path(job.dir).parent_path(), ecs).available;
            const uint64_t need = disk_store_->batch_appended_bytes() + 64ull * 1024 * 1024;
            if (!ecs && free < need) {
                disk_store_->abort_commit();
                // evict across the shared pool; drop_evicted needs a cleared
                // queued batch (the abort above frees it)
                std::vector<std::array<uint8_t, 32>> evicted;
                disk_prune_dpc(job.dir, &evicted, need);
                if (!evicted.empty()) {
                    std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
                    if (disk_store_) {
                        disk_store_->drop_evicted(evicted);
                    }
                }
                return false;
            }
        }

        prompt_cache_chunk_store::state_ref st;
        dpc_sha256_raw((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token), st.fp.data());
        st.n_tokens = tokens.size();
        st.chunks   = std::move(ids);
        if (!disk_store_->put_state(st) || !disk_store_->commit()) {
            disk_store_->abort_commit();
            return false;
        }
        return true;
    };

    if (!write_batch()) {
        disk_store_->compact();
        if (!write_batch()) {
            SRV_WRN(" - disk prompt cache: chunk store write failed after compact in '%s', using .dpc writer\n", job.dir.c_str());
            return false;
        }
    }

    disk_write_meta_file(job.dir, job.fingerprint);

    // store index so a later boot can build a restore candidate for this
    // state without scanning the store log. the split (main vs draft chunk ids)
    // rides in the sidecar's len_* fields: a (0,0) pair means a legacy pre-split
    // entry whose chunks are all main. v2 appends the checkpoints so a store
    // restore can resume instead of re-processing the whole prompt.
    if (!disk_write_token_index(job.dir, dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token)),
                                tokens, slices.size(), drft_slices.size(), state.prompt.checkpoints)) {
        std::array<uint8_t, 32> fp{};
        dpc_sha256_raw((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token), fp.data());
        std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
        if (disk_store_) {
            disk_store_->drop_state(fp);
        }
        return false;
    }

    const size_t state_bytes = state.size();
    SRV_INF(" - disk prompt cache: wrote through state '%zu' tokens to chunk store (%.3f MiB, %zu + %zu slices)\n",
            tokens.size(), (double) state_bytes / (1024.0 * 1024.0), slices.size(), drft_slices.size());
    return true;
}

void server_prompt_cache_disk::disk_thread_main() {
    while (true) {
        disk_write_job job;
        {
            std::unique_lock<std::mutex> lock(disk_jobs_mtx_);
            disk_jobs_cv_.wait(lock, [this] { return disk_thread_stop_ || !disk_jobs_.empty(); });
            if (disk_thread_stop_ && disk_jobs_.empty()) {
                return;
            }
            job = std::move(disk_jobs_.front());
            disk_jobs_.pop_front();
        }

        switch (job.kind) {
        case disk_write_job::type::state:
            disk_run_state_job(job);
            break;
        case disk_write_job::type::sidecar:
            disk_run_sidecar_job(job);
            break;
        }
    }
}

void server_prompt_cache_disk::disk_run_state_job(disk_write_job & job) {
    namespace fs = std::filesystem;

    const auto & dir  = job.dir;
    auto & state      = *job.state;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        fs::create_directories(dir, ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: failed to create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
            return;
        }
    }

    const llama_tokens tokens = state.prompt.tokens.get_text_tokens();

    // chunk-store write-through: slice the KV blob into fixed cells and commit
    // one append-only batch. a sliceable state + usable store replaces the .dpc
    // write; an unslicable blob or a failed batch keeps the legacy writer below
    // (its own disk-full guard then decides)
    if (disk_store_write(job, tokens)) {
        // register the new .dts sidecar so load_from_disk_tier can match it
        // without waiting for a restart (load_disk_index only runs at boot)
        {
            const std::string fp_hex = dpc_sha256_hex((const uint8_t *) tokens.data(), tokens.size() * sizeof(llama_token));
            const std::string fname  = fp_hex + ".dts";
            const std::string fpath  = (fs::path(dir) / fname).string();
            std::error_code ec_size;
            const auto sz = fs::file_size(fpath, ec_size);
            if (!ec_size) {
                std::lock_guard<std::mutex> lock(disk_index_mtx_);
                disk_entries.push_back({ fname, fpath, server_tokens(tokens, false), (size_t) sz });
                disk_index_valid = true;
            }
        }

        // enforce pool budget after a store write; track .dts evictions so the
        // chunk store can drop the corresponding states
        std::vector<std::array<uint8_t, 32>> evicted;
        disk_prune_dpc(dir, &evicted);
        {
            std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
            if (disk_store_) {
                disk_store_->drop_evicted(evicted);
                if (evicted.empty()) {
                    // no .dts eviction occurred: check size-based trigger
                    if (disk_store_->should_compact(disk_max_size)) {
                        disk_store_->compact();
                    }
                } else {
                    // eviction path: compact_if_stale checks dead/live and tomb/live
                    disk_store_->compact_if_stale();
                }
            }
        }

        // keep the in-memory index in sync after prune
        {
            std::error_code ec_x;
            std::lock_guard<std::mutex> lock(disk_index_mtx_);
            disk_entries.erase(
                std::remove_if(disk_entries.begin(), disk_entries.end(),
                    [&](const disk_entry & e) { return !fs::exists(e.path, ec_x); }),
                disk_entries.end());
        }

        return;
    }

    // v4 only: chunk-store write-through is required. the legacy .dpc fallback is
    // no longer supported — if the chunk store write failed, the entry is dropped
    SRV_WRN("v4 only: chunk store write failed, dropping state%s", "");
}

void server_prompt_cache_disk::disk_run_sidecar_job(disk_write_job & job) {
    namespace fs = std::filesystem;

    const std::string tmp = job.path + ".tmp";

    FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        SRV_WRN(" - disk prompt cache: cannot open sidecar '%s'\n", tmp.c_str());
        return;
    }
    const bool ok = !job.bytes.empty() && fwrite(job.bytes.data(), 1, job.bytes.size(), f) == job.bytes.size();
    fclose(f);

    if (!ok) {
        SRV_WRN(" - disk prompt cache: write to sidecar '%s' failed\n", tmp.c_str());
        std::error_code ec;
        fs::remove(tmp, ec);
        return;
    }

    std::error_code ec;
    fs::rename(tmp, job.path, ec);
    if (ec) {
        SRV_WRN(" - disk prompt cache: rename sidecar to '%s' failed: %s\n", job.path.c_str(), ec.message().c_str());
    }
}

bool server_prompt_cache::load_from_disk_tier(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    namespace fs = std::filesystem;

    // scan the index under lock; file I/O happens outside the lock so the writer
    // thread can keep pushing entries while a slow restore runs. retry when the
    // best entry's file went stale so a dead LRU head cannot shadow live ones
    while (true) {
        std::string best_path;
        std::string best_fname;
        bool restored_from_store = false; // set when reassembled from cache.dcp chunks
        float f_keep_best = 0.0f;
        float f_sim_best  = 0.0f;

        {
            std::lock_guard<std::mutex> lock(disk_index_mtx_);
            if (!disk_index_valid || disk_entries.empty()) {
                return true;
            }

            // selection mirrors the in-RAM search in load()
            const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);

            f_keep_best = prompt.tokens.size() > 0 ? float(lcp_base) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
            f_sim_best  = float(lcp_base) / tokens_new.size();

            std::vector<std::pair<size_t, size_t>> cands;
            cands.reserve(disk_entries.size());

            for (size_t i = 0; i < disk_entries.size(); ++i) {
                const auto & entry = disk_entries[i];

                const int lcp_cur = entry.tokens.get_common_prefix(tokens_new);

                const float f_keep_cur = entry.tokens.size() > 0 ? float(lcp_cur) / entry.tokens.size() : -1.0f;
                const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

                SRV_TRC(" - disk prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n",
                        entry.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

                cands.push_back({ entry.tokens.size(), (size_t) lcp_cur });
            }

            const size_t idx_best = select_cache_candidate(cands, (size_t) lcp_base);

            if (idx_best == SIZE_MAX) {
                return true;
            }

            const float f_keep_cur = float(cands[idx_best].second) / cands[idx_best].first;
            const float f_sim_cur  = float(cands[idx_best].second) / tokens_new.size();

            best_path  = disk_entries[idx_best].path;
            best_fname = disk_entries[idx_best].fname;
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
        }

        SRV_TRC(" - disk prompt cache: found entry '%s' with f_keep = %.3f, f_sim = %.3f\n", best_fname.c_str(), f_keep_best, f_sim_best);

        server_prompt_cache_state state;
        {
            FILE * f = fopen(best_path.c_str(), "rb");
            if (f == nullptr) {
                SRV_WRN(" - disk prompt cache: cannot open '%s', dropping stale entry\n", best_path.c_str());

                // prune the dead entry from the in-memory index and re-select
                std::lock_guard<std::mutex> lock(disk_index_mtx_);
                disk_entries.erase(
                    std::remove_if(disk_entries.begin(), disk_entries.end(),
                        [&](const disk_entry & e) { return e.path == best_path; }),
                    disk_entries.end());

                continue;
            }

            const auto drop_stale = [&]() {
                std::lock_guard<std::mutex> lock(disk_index_mtx_);
                disk_entries.erase(
                    std::remove_if(disk_entries.begin(), disk_entries.end(),
                        [&](const disk_entry & e) { return e.path == best_path; }),
                    disk_entries.end());
            };

            // store-backed entry: the KV lives in cache.dcp as lineage-chained
            // cells. reassemble the blob from its chunks; a fresh full restore
            // starts at position 0 and reuses via LCP. v4 keeps only the 44-byte
            // checkpoint metadata headers and reassembles the payloads from the
            // REGION_CKPT chunks trailing the main/draft ids.
            dts_header hdr;
            if (!disk_parse_dts_header(f, hdr)) {
                SRV_ERR("load_from_disk_tier: disk_parse_dts_header failed for '%s'\n", best_fname.c_str());
                fclose(f);
                return false;
            }

                llama_tokens dts_tokens(hdr.n_tokens);
                if (hdr.n_tokens > 0 && fread(dts_tokens.data(), sizeof(llama_token), hdr.n_tokens, f) != hdr.n_tokens) {
                    SRV_ERR("load_from_disk_tier: token fread failed for '%s' (n_tokens=%u)\n", best_fname.c_str(), hdr.n_tokens);
                    fclose(f);
                    return false;
                }

                // v4 = metadata headers only
                if (!disk_read_ckpt_meta(f, hdr.n_ckpt, state.prompt.checkpoints)) {
                    SRV_ERR("load_from_disk_tier: disk_read_ckpt_meta failed for '%s' (n_ckpt=%u)\n", best_fname.c_str(), hdr.n_ckpt);
                    fclose(f);
                    return false;
                }
                fclose(f);

                // ensure the chunk store is open; the same pool instance serves
                // the writer thread, and io_mtx_ guards the concurrent reads
                // (a .dts index entry with the store disabled is stale - the
                // boot rollback prune normally removes both, drop it here too)
                const uint64_t base = fingerprint_lineage(disk_fingerprint_);
                {
                    std::lock_guard<std::mutex> store_lock(disk_store_mtx_);
                    if (!prompt_cache_chunk_store::store_enabled()) {
                        drop_stale();
                        continue;
                    }
                    if (!disk_store_) {
                        disk_store_ = std::make_unique<prompt_cache_chunk_store>();
                    }
                    if (!disk_store_->is_open() &&
                        !disk_store_->open((fs::path(best_path).parent_path()).string(), base)) {
                        SRV_WRN(" - disk prompt cache: chunk store unavailable, dropping stale entry '%s'\n", best_path.c_str());

                        drop_stale();
                        continue;
                    }
                }

                std::array<uint8_t, 32> fp;
                dpc_sha256_raw((const uint8_t *) dts_tokens.data(), dts_tokens.size() * sizeof(llama_token), fp.data());

                prompt_cache_chunk_store::state_ref st;
                if (!disk_store_->get_state(fp, st)) {
                    SRV_ERR("load_from_disk_tier: chunk store get_state failed for '%s' (fp mismatch - likely old index with garbage tokens)\n", best_fname.c_str());
                    drop_stale();
                    continue;
                }
                if (st.n_tokens != (uint64_t) dts_tokens.size()) {
                    SRV_ERR("load_from_disk_tier: chunk store n_tokens mismatch for '%s': dts_tokens=%zu st.n_tokens=%llu\n", best_fname.c_str(), dts_tokens.size(), (unsigned long long) st.n_tokens);
                    drop_stale();
                    continue;
                }

                // chunk ids list the slices in the order they were written
                // (disk_store_write): main, then draft, then (v3 only) the
                // REGION_CKPT chunks for the checkpoints in sidecar order. the
                // sidecar's len_main/len_drft carry the main/draft split; a
                // (0,0) pair reads as a legacy pre-split entry (all main)
                {
                    const uint64_t total = (uint64_t) st.chunks.size();
                    const bool legacy = (hdr.len_main == 0 && hdr.len_drft == 0);
                    const uint64_t n_main = legacy ? total : hdr.len_main;
                    const uint64_t n_drft = legacy ? 0 : hdr.len_drft;
                    // v3: the trailing chunk ids must match, in sidecar order,
                    // the expected REGION_CKPT count for the checkpoint payloads
                    const uint64_t n_ckpt_ids = hdr.version == DTS_VERSION
                        ? disk_ckpt_chunk_count(hdr.n_ckpt, state.prompt.checkpoints) : 0;
                    if (n_main + n_drft + n_ckpt_ids != total) {
                        SRV_ERR("failed to load prompt state from disk ('%s'): split mismatch\n", best_fname.c_str());
                        return false;
                    }
                    try {
                        state.data.main.clear();
                        state.data.drft.clear();
                        for (uint64_t i = 0; i < n_main; ++i) {
                            std::vector<uint8_t> cell;
                            if (!disk_store_->get_chunk(st.chunks[i], cell)) {
                                SRV_ERR("failed to restore chunk %llu for '%s'\n", (unsigned long long) st.chunks[i], best_fname.c_str());
                                return false;
                            }
                            state.data.main.insert(state.data.main.end(), cell.begin(), cell.end());
                        }
                        for (uint64_t i = n_main; i < n_main + n_drft; ++i) {
                            std::vector<uint8_t> cell;
                            if (!disk_store_->get_chunk(st.chunks[i], cell)) {
                                SRV_ERR("failed to restore draft chunk %llu for '%s'\n", (unsigned long long) st.chunks[i], best_fname.c_str());
                                return false;
                            }
                            state.data.drft.insert(state.data.drft.end(), cell.begin(), cell.end());
                        }
                        // v3: reassemble each checkpoint stream from its
                        // REGION_CKPT chunks (written in stream order 0,1,2,
                        // each tiled at MAX_REC_LEN). the meta reader sized the
                        // payload vectors, so a full pass must reproduce them
                        // byte-for-byte; append into the resized buffers via a
                        // write cursor rather than re-inserting
                        if (hdr.version == DTS_VERSION && n_ckpt_ids > 0) {
                            uint64_t id_i = n_main + n_drft;
                            for (auto & ckpt : state.prompt.checkpoints) {
                                std::vector<uint8_t> * streams[3] = { &ckpt.data_tgt, &ckpt.data_dft, &ckpt.data_spec };
                                for (auto * stream : streams) {
                                    uint64_t off = 0;
                                    while (off < stream->size() && id_i < total) {
                                        std::vector<uint8_t> cell;
                                        if (!disk_store_->get_chunk(st.chunks[id_i++], cell)) {
                                            SRV_ERR("failed to restore checkpoint chunk %llu for '%s'\n",
                                                    (unsigned long long) st.chunks[0], best_fname.c_str());
                                            return false;
                                        }
                                        memcpy(stream->data() + off, cell.data(), cell.size());
                                        off += cell.size();
                                    }
                                    if (off != stream->size()) {
                                        SRV_ERR("failed to restore prompt state from disk ('%s'): checkpoint stream underfilled\n", best_fname.c_str());
                                        return false;
                                    }
                                }
                            }
                            if (id_i != total) {
                                SRV_ERR("failed to restore prompt state from disk ('%s'): checkpoint chunks overrun\n", best_fname.c_str());
                                return false;
                            }
                        }
                    } catch (const std::bad_alloc & e) {
                        SRV_ERR("failed to allocate memory for prompt state from disk ('%s'): %s\n", best_fname.c_str(), e.what());
                        return false;
                    }

                    // a compact can renumber chunk ids while the get_chunk loops
                    // above ran; the captured list would then reassemble a
                    // wrong-but-valid blob. re-fetch the state and confirm the id
                    // list is unchanged, then re-walk both blobs with the same
                    // strict pos==i contract the writer used before restore_state
                    // feeds them to the memory
                    {
                        prompt_cache_chunk_store::state_ref st_now;
                        const bool present = disk_store_->get_state(fp, st_now);
                        bool same_ids = present && st_now.n_tokens == st.n_tokens &&
                                        st_now.chunks.size() == st.chunks.size();
                        if (same_ids) {
                            for (size_t i = 0; i < st.chunks.size() && same_ids; ++i) {
                                same_ids = st.chunks[i] == st_now.chunks[i];
                            }
                        }

                        std::vector<state_blob_slice> chk;
                        const bool main_ok = slice_state_blob(state.data.main.data(), state.data.main.size(),
                                                              (size_t) dts_tokens.size(), chk);
                        bool drft_ok = true;
                        if (!legacy && n_drft > 0) {
                            const uint32_t drft_cells = state.data.drft.size() >= 16
                                ? cc_io::get_u32(state.data.drft.data() + 12) : 0;
                            std::vector<state_blob_slice> chk_d;
                            drft_ok = drft_cells > 0 && drft_cells <= (uint32_t) dts_tokens.size() &&
                                      slice_state_blob(state.data.drft.data(), state.data.drft.size(), drft_cells, chk_d,
                                                       (uint8_t) prompt_cache_chunk_store::REGION_DRAFT);
                        }

                        if (present && !same_ids) {
                            // ids remapped while restoring: retry against the
                            // fresh id list instead of dropping a live entry
                            SRV_WRN(" - disk prompt cache: entry '%s' churn detected during restore (gen %llu, %zu -> %zu ids), retrying\n",
                                    best_fname.c_str(), (unsigned long long) disk_store_->generation(),
                                    st.chunks.size(), st_now.chunks.size());
                            continue;
                        }
                        if (!main_ok || !drft_ok) {
                            SRV_ERR(" - disk prompt cache: entry '%s' failed blob self-check (gen %llu, main_ok=%d, drft_ok=%d), dropping stale entry\n",
                                    best_fname.c_str(), (unsigned long long) disk_store_->generation(),
                                    (int) main_ok, (int) drft_ok);
                            drop_stale();
                            continue;
                        }
                    }
                }
                state.prompt.tokens = server_tokens(dts_tokens, false);
                restored_from_store = true;
            }

        // promote into the RAM tier when it fits the budget, otherwise restore directly from disk
        const size_t state_bytes = state.size();
        const bool can_promote = limit_size == 0 || size() + state_bytes <= limit_size;

        if (can_promote) {
            states.push_back(std::move(state));

            if (!restore_state(prompt, states.back(), ctx_tgt, ctx_dft, id_slot)) {
                states.pop_back();

                return false;
            }

            // consume like a RAM hit: the state now lives in the slot
            states.pop_back();

            SRV_INF(" - disk prompt cache: promoted entry '%s' into RAM (%.3f MiB)\n", best_fname.c_str(), state_bytes / (1024.0 * 1024.0));
        } else {
            if (!restore_state(prompt, state, ctx_tgt, ctx_dft, id_slot)) {
                return false;
            }

            SRV_INF(" - disk prompt cache: restored entry '%s' directly from disk (%.3f MiB, RAM budget exhausted)\n", best_fname.c_str(), state_bytes / (1024.0 * 1024.0));
        }

        if (restored_from_store) {
            // probe the memory right after a chunk-store restore: a pos_min at
            // the threshold suggests the reassembled blob carried wrong cells,
            // while a healthy pos_min points at post-restore corruption
            const llama_pos p_attn = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id_slot);
            const llama_pos p_drft = ctx_dft ? llama_memory_seq_pos_min(llama_get_memory(ctx_dft), id_slot) : -1;
            SRV_TRC(" - disk prompt cache: store-restored '%s' attn pos_min = %d, drft pos_min = %d\n",
                    best_fname.c_str(), (int) p_attn, (int) p_drft);
        }

        // successful restore refreshes the LRU order so the prune prefers cold entries
        std::error_code ec;
        fs::last_write_time(best_path, fs::file_time_type::clock::now(), ec);
        if (ec) {
            SRV_WRN(" - disk prompt cache: cannot touch '%s': %s\n", best_path.c_str(), ec.message().c_str());
        }

        return true;
    }
}

size_t server_prompt_cache_disk::load_disk_index(const std::string & dir, const std::string & fingerprint) {
    namespace fs = std::filesystem;

    disk_dir_         = dir;
    disk_fingerprint_ = fingerprint;

    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return 0;
    }

    // store disabled: one-pass rollback on the next boot. remove the store
    // artifacts (.dcp/.dci/.dts/cache.lock) so the tree reverts to the legacy
    // .dpc/.ccn layout; never touch those. idempotent - a store-enabled boot
    // skips this entirely
    if (!prompt_cache_chunk_store::store_enabled()) {
        for (auto & entry : fs::directory_iterator(dir, ec)) {
            const std::string name = entry.path().filename().string();
            if (entry.path().extension() == ".dcp" || entry.path().extension() == ".dci" ||
                entry.path().extension() == ".dts" || name == "cache.lock") {
                fs::remove(entry.path(), ec);
            }
        }
        SRV_WRN(" - disk prompt cache: chunk store disabled, pruned store files in '%s' (rollback)\n", dir.c_str());
    }

    std::string stored;
    {
        std::ifstream in((fs::path(dir) / "meta").string(), std::ios::binary);
        if (!in) {
            SRV_INF(" - disk prompt cache: no meta in '%s', skipping load\n", dir.c_str());
            return 0;
        }
        stored.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        // strip trailing newlines (older saves wrote a trailing '\n')
        while (!stored.empty() && (stored.back() == '\n' || stored.back() == '\r')) {
            stored.pop_back();
        }
    }

    if (stored != fingerprint) {
        SRV_INF(" - disk prompt cache: fingerprint mismatch for '%s', ignoring cached prompts\n", dir.c_str());
        return 0;
    }

    std::vector<fs::path> files;
    for (auto & entry : fs::directory_iterator(dir, ec)) {
        if (entry.path().extension() == ".dts") {
            files.push_back(entry.path());
        }
    }

    // deterministic order
    std::sort(files.begin(), files.end());

    // read the token prefix and tracked sizes only; the KV blobs stay on disk
    // until a matching entry is actually restored. build the index locally and
    // swap it in under the lock so the writer thread never sees a partial index
    std::vector<disk_entry> new_entries;

    size_t n_indexed = 0;
    for (const auto & path : files) {
        FILE * f = fopen(path.string().c_str(), "rb");
        if (f == nullptr) {
            continue;
        }

        uint32_t magic    = 0;
        uint32_t version  = 0;
        uint32_t n_tokens = 0;
        uint64_t len_main = 0;
        uint64_t len_drft = 0;
        uint32_t n_ckpt   = 0;
        uint64_t last_access = 0;

        bool ok =
            fread(&magic, 4, 1, f) == 1 && fread(&version, 4, 1, f) == 1 &&
            fread(&n_tokens, 4, 1, f) == 1 && fread(&len_main, 8, 1, f) == 1 &&
            fread(&len_drft, 8, 1, f) == 1 && fread(&n_ckpt, 4, 1, f) == 1 &&
            fread(&last_access, 8, 1, f) == 1 &&
            magic == DTS_MAGIC && version == DTS_VERSION &&
            n_tokens <= DPC_MAX_TOKENS && n_ckpt <= DPC_MAX_CHECKPOINTS;

        llama_tokens tokens(n_tokens);
        ok = ok && (n_tokens == 0 || fread(tokens.data(), sizeof(llama_token), n_tokens, f) == n_tokens);
        fclose(f);

        if (!ok) {
            SRV_WRN(" - disk prompt cache: ignoring corrupt entry '%s'\n", path.filename().string().c_str());
            continue;
        }

        new_entries.push_back({
            /*.fname      =*/ path.filename().string(),
            /*.path       =*/ path.string(),
            /*.tokens     =*/ server_tokens(tokens, false),
            /*.byte_size  =*/ (size_t) fs::file_size(path, ec),
        });

        n_indexed++;
    }

    {
        std::lock_guard<std::mutex> lock(disk_index_mtx_);
        disk_entries     = std::move(new_entries);
        disk_index_valid = true;
    }

    return n_indexed;
}
