#include "server-task.h"
#include "server-prompt-cache-store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define cc_fileno _fileno
#else
#include <unistd.h>
#include <sys/file.h>
#define cc_fileno fileno
#endif

//
// TurboQuant chunked disk prompt cache
//
// replaces the "<sha256>.dpc" per-state files with one append-only cache.dcp
// record log shared by every state in the directory:
//
//   cache.dcp   record log (chunks + state refs + tombstones), generation
//               tagged so a torn tail - a partial append or an uncommitted
//               batch - is detected and truncated at open
//   cache.dci   index snapshot (chunk offsets + state references); a cache of
//               the authoritative .dcp, always rebuildable from a full scan
//   cache.lock  advisory single-writer lock (flock)
//
// a state blob is sliced by the save path into fixed-cell (region, layer)
// sheets; identical sheets dedup to an existing chunk. state records map a
// sha256 token fingerprint to the chunk ids that rebuild it in byte order.
//
// record layout (all little-endian):
//   u16 kind   (1 = chunk, 2 = state, 3 = tombstone)
//   u16 region (REGION_*)
//   u32 layer
//   u32 cell_lo
//   u32 cell_hi
//   u64 lineage_epoch
//   u64 gen    (commit generation the record belongs to)
//   u64 len    (payload size)
//   u32 crc32c (of the record header bytes [0..40) + payload)
//   u32 pad
// then payload, then zero-padded to the next 8-byte boundary.
//
// the cache.dcp header is a fixed 24 bytes (magic, version, base lineage,
// generation); a commit rewrites the generation after fsyncing the records.
// records tagged gen > header gen belong to a batch that never committed and
// are discarded at open. the base lineage is a fingerprint-derived seed; a
// different base resets the store (same semantics as the .dpc/.ccn wipe).

namespace {

constexpr uint32_t CHUNK_REC_HDR = 48;
constexpr uint16_t CHUNK_REC_CHUNK    = 1;
constexpr uint16_t CHUNK_REC_STATE    = 2;   // payload: fp[32] + n_tokens u64 + ids u64[n]
constexpr uint16_t CHUNK_REC_TOMB     = 3;   // payload: fp[32]

// crc32c = Castagnoli, reflected
uint32_t g_crc32c_table[256];
bool     g_crc32c_ready = false;

void crc32c_init_table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int b = 0; b < 8; b++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78u : 0);
        }
        g_crc32c_table[i] = crc;
    }
    g_crc32c_ready = true;
}

uint32_t crc32c(const uint8_t * data, size_t len) {
    if (!g_crc32c_ready) {
        crc32c_init_table();
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ g_crc32c_table[(crc ^ data[i]) & 0xff];
    }
    return crc ^ 0xFFFFFFFFu;
}

uint32_t crc32c_update(uint32_t crc, const uint8_t * data, size_t len) {
    if (!g_crc32c_ready) {
        crc32c_init_table();
    }
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ g_crc32c_table[(crc ^ data[i]) & 0xff];
    }
    return crc;
}

} // namespace

struct prompt_cache_chunk_store::impl {
    std::string dir_;
    std::string dcp_path_;
    std::string dci_path_;
    std::string lock_path_;

    FILE * f_       = nullptr;   // cache.dcp
    int    lock_fd_ = -1;        // cache.lock

    uint64_t base_lineage_    = 0;
    uint64_t committed_gen_   = 0;   // generation in the .dcp header
    uint64_t queued_gen_      = 0;   // 0 = no open batch
    uint64_t batch_start_     = 0;   // file offset where the open batch appends
    size_t   batch_chunks_    = 0;   // chunks_.size() at batch start
    uint64_t batch_appended_  = 0;   // bytes this batch physically appended
    std::vector<std::array<uint8_t, 32>> batch_states_;
    std::vector<std::array<uint8_t, 32>> batch_tombs_;

    struct chunk {
        chunk_key key;
        uint64_t  offset = 0;   // payload offset in cache.dcp
        uint64_t  len    = 0;
        uint32_t  crc    = 0;
        uint32_t  refs   = 0;   // live state references
    };
    std::vector<chunk> chunks_;
    std::map<std::string, std::vector<uint64_t>> states_;  // fp-str -> ids, byte order
    std::map<std::string, uint64_t> state_tokens_;          // fp-str -> n_tokens
    std::map<chunk_key, uint64_t>  by_key_;                 // dedup oracle -> chunk id

    uint64_t n_tombs_ = 0;   // persistent tombstone count in .dci header

    bool writable_   = false;
    bool want_dedup_ = true;
    bool dci_valid_  = false;
    bool close_log_  = false;   // header rewrite failed: stop writes

    std::mutex io_mtx_;

    impl() {
        want_dedup_ = store_enabled();
    }

    ~impl() = default;

    static size_t rec_size(size_t len) {
        return CHUNK_REC_HDR + ((len + 7) & ~(size_t) 7);
    }

    bool file_size_of(const std::string & path, uint64_t & out) const {
        std::error_code ec;
        const uintmax_t sz = std::filesystem::file_size(path, ec);
        if (ec) {
            return false;
        }
        out = sz;
        return true;
    }

    // read len bytes at offset into buf (buffered, position-independent)
    bool pread_at(int64_t off, uint8_t * buf, size_t len) {
        if (cc_fileno(f_) < 0) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(f_, off, SEEK_SET) != 0) {
            return false;
        }
        return fread(buf, 1, len, f_) == len;
#else
        const ssize_t n = pread(cc_fileno(f_), buf, len, (off_t) off);
        return n == (ssize_t) len;
#endif
    }

    // append payload at the current end of file; returns [0] logical offset
    bool append_at_end(const uint8_t * data, size_t len, uint64_t & off) {
        if (cc_fileno(f_) < 0) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(f_, 0, SEEK_END) != 0) {
            return false;
        }
        off = (uint64_t) _ftelli64(f_);
        if (fwrite(data, 1, len, f_) != len) {
            return false;
        }
        return fflush(f_) == 0;
#else
        off = (uint64_t) lseek(cc_fileno(f_), 0, SEEK_END);
        const ssize_t n = pwrite(cc_fileno(f_), data, len, (off_t) off);
        return n == (ssize_t) len;
#endif
    }

    bool sync_file() {
#ifdef _WIN32
        return _commit(cc_fileno(f_)) == 0;
#else
        return fsync(cc_fileno(f_)) == 0;
#endif
    }

    bool truncate_to(uint64_t size) {
#ifdef _WIN32
        return _chsize_s(cc_fileno(f_), (long long) size) == 0;
#else
        return ftruncate(cc_fileno(f_), (off_t) size) == 0;
#endif
    }

    bool set_header_gen(uint64_t gen) {
        uint8_t hdr[24];
        memset(hdr, 0, sizeof(hdr));
        cc_io::put_u32(hdr,      DCP_MAGIC);
        cc_io::put_u32(hdr + 4,  FORMAT_VER);
        cc_io::put_u64(hdr + 8,  base_lineage_);
        cc_io::put_u64(hdr + 16, gen);
        return append_rewrite_hdr(hdr);
    }

    bool append_rewrite_hdr(const uint8_t * hdr) {
        if (cc_fileno(f_) < 0) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(f_, 0, SEEK_SET) != 0 || fwrite(hdr, 1, 24, f_) != 24 || fflush(f_) != 0) {
            close_log_ = true;
            return false;
        }
#else
        const ssize_t wrote = pwrite(cc_fileno(f_), hdr, 24, 0);
        if (wrote != 24) {
            close_log_ = true;   // header rewrite failed, stop writes
            return false;
        }
#endif
        return true;
    }

    // needs io_mtx_; writes one record at the end of the file
    bool append_record(uint16_t kind, const chunk_key & key, uint64_t gen,
                       const uint8_t * payload, size_t len, uint64_t & payload_off) {
        if (close_log_) {
            return false;
        }
        std::vector<uint8_t> hdr(CHUNK_REC_HDR);
        cc_io::put_u16(hdr.data(),       kind);
        cc_io::put_u16(hdr.data() + 2,   key.region);
        cc_io::put_u32(hdr.data() + 4,   key.layer);
        cc_io::put_u32(hdr.data() + 8,   key.cell_lo);
        cc_io::put_u32(hdr.data() + 12,  key.cell_hi);
        cc_io::put_u64(hdr.data() + 16,  key.lineage_epoch);
        cc_io::put_u64(hdr.data() + 24,  gen);
        cc_io::put_u64(hdr.data() + 32,  len);
        cc_io::put_u32(hdr.data() + 40,  crc32c(hdr.data(), 40) ^ crc32c(payload, len));
        cc_io::put_u32(hdr.data() + 44,  0);

        std::vector<uint8_t> rec(CHUNK_REC_HDR + ((len + 7) & ~(size_t) 7));
        memcpy(rec.data(), hdr.data(), CHUNK_REC_HDR);
        memcpy(rec.data() + CHUNK_REC_HDR, payload, len);
        memset(rec.data() + CHUNK_REC_HDR + len, 0, rec.size() - CHUNK_REC_HDR - len);

        uint64_t off = 0;
        if (!append_at_end(rec.data(), rec.size(), off)) {
            return false;
        }
        payload_off = off + CHUNK_REC_HDR;
        return true;
    }

    std::string fp_str(const std::array<uint8_t, 32> & fp) {
        return std::string((const char *) fp.data(), 32);
    }

    void recompute_refs() {
        for (auto & c : chunks_) {
            c.refs = 0;
        }
        for (const auto & it : states_) {
            for (uint64_t id : it.second) {
                if (id < chunks_.size()) {
                    chunks_[id].refs++;
                }
            }
        }
    }

    // needs io_mtx_; rebuild all in-memory state from a full .dcp scan.
    // returns false when the file is corrupt beyond a resettable state.
    bool rebuild_from_scan() {
        chunks_.clear();
        states_.clear();
        state_tokens_.clear();
        by_key_.clear();
        n_tombs_ = 0;

        // read and verify the header first
        uint8_t hdr[24];
        if (!pread_at(0, hdr, 24)) {
            return false;
        }
        if (cc_io::get_u32(hdr) != DCP_MAGIC || cc_io::get_u32(hdr + 4) != FORMAT_VER) {
            return false;
        }

        uint64_t filesz = 0;
        if (!file_size_of(dcp_path_, filesz)) {
            return false;
        }

        int64_t off = 24;
        while ((uint64_t) off + CHUNK_REC_HDR <= filesz) {
            uint8_t rh[CHUNK_REC_HDR];
            if (!pread_at(off, rh, CHUNK_REC_HDR)) {
                break;
            }
            const uint16_t kind   = cc_io::get_u16(rh);
            const uint16_t region = cc_io::get_u16(rh + 2);
            const uint32_t layer  = cc_io::get_u32(rh + 4);
            const uint32_t cell_lo = cc_io::get_u32(rh + 8);
            const uint32_t cell_hi = cc_io::get_u32(rh + 12);
            const uint64_t lineage = cc_io::get_u64(rh + 16);
            const uint64_t gen     = cc_io::get_u64(rh + 24);
            const uint64_t len     = cc_io::get_u64(rh + 32);
            const uint32_t crc     = cc_io::get_u32(rh + 40);

            if (kind != CHUNK_REC_CHUNK && kind != CHUNK_REC_STATE && kind != CHUNK_REC_TOMB) {
                break;   // torn tail
            }
            if (len > MAX_REC_LEN) {
                break;
            }
            const int64_t p_off = off + CHUNK_REC_HDR;
            if ((uint64_t) p_off + len > filesz) {
                break;   // short payload
            }
            std::vector<uint8_t> payload(len);
            if (len > 0 && !pread_at(p_off, payload.data(), len)) {
                break;
            }
            if ((crc32c(rh, 40) ^ crc32c(payload.data(), len)) != crc) {
                break;   // torn/corrupt record
            }

            // records from a batch that never committed (gen > header gen)
            if (gen > committed_gen_) {
                break;
            }

            if (kind == CHUNK_REC_CHUNK) {
                chunk c;
                c.key.region        = (uint8_t) region;
                c.key.layer         = (uint16_t) layer;
                c.key.cell_lo       = cell_lo;
                c.key.cell_hi       = cell_hi;
                c.key.lineage_epoch = lineage;
                c.offset            = (uint64_t) p_off;
                c.len               = len;
                c.crc               = crc;
                c.refs              = 0;
                chunks_.push_back(c);
                by_key_[c.key] = chunks_.size() - 1;
            } else if (kind == CHUNK_REC_STATE) {
                if (payload.size() < 32 + 8 + 8) {
                    break;
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), payload.data(), 32);
                const uint64_t n_tokens = cc_io::get_u64(payload.data() + 32);
                const uint64_t n        = cc_io::get_u64(payload.data() + 40);
                if (n > (payload.size() - 48) / 8) {
                    break;
                }
                std::vector<uint64_t> ids(n);
                for (uint64_t i = 0; i < n; i++) {
                    ids[i] = cc_io::get_u64(payload.data() + 48 + 8 * i);
                    if (ids[i] >= chunks_.size()) {
                        break;   // reference to a later/absent chunk
                    }
                }
                const std::string fp_s = fp_str(fp);
                states_[fp_s] = std::move(ids);
                state_tokens_[fp_s] = n_tokens;
            } else { // tombstone
                if (payload.size() < 32) {
                    break;
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), payload.data(), 32);
                states_.erase(fp_str(fp));
                state_tokens_.erase(fp_str(fp));
                n_tombs_++;
            }

            off += (int64_t) rec_size(len);
        }

        // a clean scan ends exactly on a record boundary; anything past the
        // last good record is garbage (torn append or torn uncommitted batch)
        if (writable_ && (uint64_t) off != filesz) {
            truncate_to((uint64_t) off);
            sync_file();
        }

        recompute_refs();
        return true;
    }

    // index snapshot on disk; valid when header + sizes match the current .dcp
    bool load_index_fast() {
        std::ifstream in(dci_path_, std::ios::binary);
        if (!in) {
            return false;
        }
        uint8_t hdr[52];
        in.read((char *) hdr, 52);
        if (!in || cc_io::get_u32(hdr) != DCI_MAGIC || cc_io::get_u32(hdr + 4) != DCI_FORMAT_VER) {
            return false;
        }
        if (cc_io::get_u64(hdr + 8)  != base_lineage_ ||
            cc_io::get_u64(hdr + 16) != committed_gen_) {
            return false;
        }
        const uint64_t dcp_size = cc_io::get_u64(hdr + 24);
        uint64_t real = 0;
        if (!file_size_of(dcp_path_, real) || real != dcp_size) {
            return false;
        }
        in.seekg(0, std::ios::end);
        const std::streamoff index_size = in.tellg();
        if (index_size < 52) {
            return false;
        }
        const uint64_t remaining = (uint64_t) index_size - 52;
        uint32_t crc = crc32c_update(0xFFFFFFFFu, hdr, 48);
        in.clear();
        in.seekg(52, std::ios::beg);
        std::array<uint8_t, 64 * 1024> crc_buf{};
        uint64_t crc_remaining = remaining;
        while (crc_remaining > 0) {
            const size_t n = (size_t) std::min<uint64_t>(crc_remaining, crc_buf.size());
            in.read((char *) crc_buf.data(), (std::streamsize) n);
            if (!in) {
                return false;
            }
            crc = crc32c_update(crc, crc_buf.data(), n);
            crc_remaining -= n;
        }
        if ((crc ^ 0xFFFFFFFFu) != cc_io::get_u32(hdr + 48)) {
            return false;
        }
        const uint32_t n_chunks = cc_io::get_u32(hdr + 32);
        const uint32_t n_states = cc_io::get_u32(hdr + 36);
        if (n_chunks > remaining / 48 || n_states > (remaining - (uint64_t) n_chunks * 48) / 48) {
            return false;
        }
        n_tombs_ = cc_io::get_u64(hdr + 40);

        chunks_.clear();
        states_.clear();
        state_tokens_.clear();
        by_key_.clear();
        in.clear();
        in.seekg(52, std::ios::beg);

        for (uint32_t i = 0; i < n_chunks; i++) {
            uint8_t e[48];
            in.read((char *) e, 48);
            if (!in) {
                return false;
            }
            if (cc_io::get_u32(e) != i) {
                return false;
            }
            chunk c;
            c.key.region        = (uint8_t) cc_io::get_u16(e + 2);
            c.key.layer         = (uint16_t) cc_io::get_u16(e + 4);
            c.key.cell_lo       = cc_io::get_u32(e + 8);
            c.key.cell_hi       = cc_io::get_u32(e + 12);
            c.key.lineage_epoch = cc_io::get_u64(e + 16);
            c.offset            = cc_io::get_u64(e + 24);
            c.len               = cc_io::get_u64(e + 32);
            c.crc               = cc_io::get_u32(e + 40);
            c.refs              = 0;
            if (c.len > MAX_REC_LEN || c.offset < CHUNK_REC_HDR + 24 ||
                c.offset > (uint64_t) std::numeric_limits<int64_t>::max() ||
                c.offset > dcp_size || c.len > dcp_size - c.offset) {
                return false;
            }
            uint8_t record[CHUNK_REC_HDR];
            if (!pread_at((int64_t) c.offset - CHUNK_REC_HDR, record, sizeof(record)) ||
                cc_io::get_u16(record) != CHUNK_REC_CHUNK ||
                cc_io::get_u16(record + 2) != c.key.region ||
                cc_io::get_u32(record + 4) != c.key.layer ||
                cc_io::get_u32(record + 8) != c.key.cell_lo ||
                cc_io::get_u32(record + 12) != c.key.cell_hi ||
                cc_io::get_u64(record + 16) != c.key.lineage_epoch ||
                cc_io::get_u64(record + 32) != c.len ||
                cc_io::get_u32(record + 40) != c.crc) {
                return false;
            }
            chunks_.push_back(c);
            by_key_[c.key] = chunks_.size() - 1;
        }

        for (uint32_t i = 0; i < n_states; i++) {
            std::array<uint8_t, 32> fp;
            in.read((char *) fp.data(), 32);
            uint8_t meta[16];
            in.read((char *) meta, 16);
            if (!in) {
                return false;
            }
            const uint64_t n_tokens = cc_io::get_u64(meta);
            const uint64_t n        = cc_io::get_u64(meta + 8);
            const std::streamoff at = in.tellg();
            if (at < 0 || n > ((uint64_t) index_size - (uint64_t) at) / 8) {
                return false;
            }
            std::vector<uint64_t> ids(n);
            for (uint64_t k = 0; k < n; k++) {
                uint8_t ib[8];
                in.read((char *) ib, 8);
                if (!in) {
                    return false;
                }
                ids[k] = cc_io::get_u64(ib);
                if (ids[k] >= n_chunks) {
                    return false;
                }
            }
            states_[fp_str(fp)] = std::move(ids);
            state_tokens_[fp_str(fp)] = n_tokens;
        }
        if (in.peek() != std::char_traits<char>::eof()) {
            return false;
        }
        recompute_refs();
        return true;
    }

    // needs io_mtx_; write cache.dci from in-memory state
    void write_index_locked() {
        const std::string tmp = dci_path_ + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        uint8_t hdr[52];
        memset(hdr, 0, sizeof(hdr));
        cc_io::put_u32(hdr,      DCI_MAGIC);
        cc_io::put_u32(hdr + 4,  DCI_FORMAT_VER);
        cc_io::put_u64(hdr + 8,  base_lineage_);
        cc_io::put_u64(hdr + 16, committed_gen_);
        uint64_t dcp_size = 0;
        file_size_of(dcp_path_, dcp_size);
        cc_io::put_u64(hdr + 24, dcp_size);
        cc_io::put_u32(hdr + 32, (uint32_t) chunks_.size());
        cc_io::put_u32(hdr + 36, (uint32_t) states_.size());
        cc_io::put_u64(hdr + 40, n_tombs_);
        out.write((const char *) hdr, 52);

        for (size_t i = 0; i < chunks_.size(); i++) {
            uint8_t e[48];
            memset(e, 0, sizeof(e));
            const chunk & c = chunks_[i];
            cc_io::put_u32(e,       (uint32_t) i);
            cc_io::put_u16(e + 2,   c.key.region);
            cc_io::put_u16(e + 4,   c.key.layer);
            cc_io::put_u32(e + 8,   c.key.cell_lo);
            cc_io::put_u32(e + 12,  c.key.cell_hi);
            cc_io::put_u64(e + 16,  c.key.lineage_epoch);
            cc_io::put_u64(e + 24,  c.offset);
            cc_io::put_u64(e + 32,  c.len);
            cc_io::put_u32(e + 40,  c.crc);
            cc_io::put_u32(e + 44,  c.refs);
            out.write((const char *) e, 48);
        }
        for (const auto & it : states_) {
            std::array<uint8_t, 32> fp;
            memcpy(fp.data(), it.first.data(), 32);
            out.write((const char *) fp.data(), 32);
            uint8_t meta[16];
            cc_io::put_u64(meta,      state_tokens_[it.first]);
            cc_io::put_u64(meta + 8,  it.second.size());
            out.write((const char *) meta, 16);
            for (uint64_t id : it.second) {
                uint8_t ib[8];
                cc_io::put_u64(ib, id);
                out.write((const char *) ib, 8);
            }
        }
        out.flush();
        if (!out) {
            return;
        }
        out.close();

        std::ifstream verify(tmp, std::ios::binary);
        uint8_t crc_hdr[48];
        verify.read((char *) crc_hdr, sizeof(crc_hdr));
        if (!verify) {
            return;
        }
        uint32_t crc = crc32c_update(0xFFFFFFFFu, crc_hdr, sizeof(crc_hdr));
        verify.seekg(52, std::ios::beg);
        std::array<uint8_t, 64 * 1024> crc_buf{};
        while (verify) {
            verify.read((char *) crc_buf.data(), crc_buf.size());
            const std::streamsize n = verify.gcount();
            if (n > 0) {
                crc = crc32c_update(crc, crc_buf.data(), (size_t) n);
            }
        }
        if (!verify.eof()) {
            return;
        }
        verify.close();
        std::fstream checksum_out(tmp, std::ios::binary | std::ios::in | std::ios::out);
        if (!checksum_out) {
            return;
        }
        uint8_t crc_bytes[4];
        cc_io::put_u32(crc_bytes, crc ^ 0xFFFFFFFFu);
        checksum_out.seekp(48, std::ios::beg);
        checksum_out.write((const char *) crc_bytes, sizeof(crc_bytes));
        checksum_out.flush();
        if (!checksum_out) {
            return;
        }
        checksum_out.close();
        std::error_code ec;
        std::filesystem::rename(tmp, dci_path_, ec);
        dci_valid_ = !ec;
    }

    // reopen the store as read-only (lock lost later); keeps existing cache.dcp
    void reopen_readonly() {
        if (f_) {
            fclose(f_);
            f_ = nullptr;
        }
        f_ = fopen(dcp_path_.c_str(), "rb");
        writable_ = false;
    }

    void open_lock() {
#ifdef _WIN32
        lock_fd_ = -1; // no advisory lock on Windows
#else
        lock_fd_ = ::open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (lock_fd_ >= 0 && ::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#endif
    }

    void close_lock() {
#ifdef _WIN32
        if (lock_fd_ >= 0) {
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#else
        if (lock_fd_ >= 0) {
            ::flock(lock_fd_, LOCK_UN);
            ::close(lock_fd_);
            lock_fd_ = -1;
        }
#endif
    }

    // roll the open batch back to the committed state (caller holds io_mtx_)
    void abort_batch_locked();

    // clear the log and rebuild an empty store (caller holds io_mtx_)
    void reset_log_locked();
};

prompt_cache_chunk_store::prompt_cache_chunk_store()
    : p_(std::make_unique<impl>()) {
}

prompt_cache_chunk_store::~prompt_cache_chunk_store() {
    close();
}

bool prompt_cache_chunk_store::open(const std::string & dir, uint64_t base_lineage) {
    p_ = std::unique_ptr<impl>(new impl());
    impl & I = *p_;

    I.dir_          = dir;
    I.dcp_path_     = (std::filesystem::path(dir) / "cache.dcp").string();
    I.dci_path_     = (std::filesystem::path(dir) / "cache.dci").string();
    I.lock_path_    = (std::filesystem::path(dir) / "cache.lock").string();
    I.base_lineage_ = base_lineage;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    I.open_lock();
    const bool have_lock = I.lock_fd_ >= 0;
    const bool exists    = std::filesystem::exists(I.dcp_path_, ec);

    // open the log for read, and for write when we hold the lock
    I.f_ = fopen(I.dcp_path_.c_str(), have_lock ? "r+b" : "rb");
    if (!I.f_ && have_lock) {
        I.f_ = fopen(I.dcp_path_.c_str(), "w+b");
    }
    if (!I.f_) {
        SRV_ERR("chunk store: cannot open %s", I.dcp_path_.c_str());
        I.close_lock();
        p_.reset();
        return false;
    }
    I.writable_ = have_lock;

    // base lineage mismatch = config/fingerprint change: reset the store;
    // an unreadable header is treated the same
    bool base_ok = false;
    if (exists) {
        uint8_t hdr[24];
        if (I.pread_at(0, hdr, 24)) {
            base_ok = cc_io::get_u32(hdr) == DCP_MAGIC &&
                      cc_io::get_u32(hdr + 4) == FORMAT_VER &&
                      cc_io::get_u64(hdr + 8) == base_lineage;
            if (base_ok) {
                I.committed_gen_ = cc_io::get_u64(hdr + 16);
            }
        }
    }
    if (exists && !base_ok && !have_lock) {
        SRV_WRN("chunk store %s: foreign, open denied", I.dcp_path_.c_str());
        I.close_lock();
        p_.reset();
        return false;
    }
    if (I.writable_ && (!exists || !base_ok)) {
        I.reset_log_locked();
    }

    // index: fast path from cache.dci, else full scan of cache.dcp
    if (!I.load_index_fast()) {
        if (!I.rebuild_from_scan()) {
            SRV_WRN("chunk store %s: corrupt log, resetting under lock", I.dcp_path_.c_str());
            if (I.writable_) {
                I.reset_log_locked();
            } else {
                I.close_lock();
                p_.reset();
                return false;
            }
        }
    }
    return true;
}

void prompt_cache_chunk_store::close() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    {
        std::lock_guard<std::mutex> lk(I.io_mtx_);
        if (I.writable_) {
            I.write_index_locked();
        }
        if (I.f_) {
            fclose(I.f_);
            I.f_ = nullptr;
        }
    }
    I.close_lock();
    p_.reset();
}

bool prompt_cache_chunk_store::store_enabled() {
    const char * v = getenv("TURBO_DISK_CHUNK_STORE");
    return !(v && strcmp(v, "0") == 0);
}

bool prompt_cache_chunk_store::is_open() const {
    return p_ && p_->f_;
}

bool prompt_cache_chunk_store::wants_dedup() const {
    return p_ ? p_->want_dedup_ : true;
}

uint64_t prompt_cache_chunk_store::batch_appended_bytes() const {
    return p_ ? p_->batch_appended_ : 0;
}

bool prompt_cache_chunk_store::begin_commit() {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return false;
    }
    // append point is the true end of the log (a batch starts after every
    // committed record, including trailing state/tombstone records)
    uint64_t batch_start = 0;
    if (!I.file_size_of(I.dcp_path_, batch_start)) {
        return false;
    }
    I.batch_start_  = batch_start;
    I.batch_chunks_ = I.chunks_.size();
    I.batch_appended_ = 0;
    I.batch_states_.clear();
    I.batch_tombs_.clear();
    I.queued_gen_ = I.committed_gen_ + 1;
    return true;
}

int64_t prompt_cache_chunk_store::put_chunk(const chunk_key & key, const uint8_t * data, size_t len) {
    if (!p_) {
        return -1;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0 || len > MAX_REC_LEN) {
        return -1;
    }
    if (I.want_dedup_) {
        const auto it = I.by_key_.find(key);
        if (it != I.by_key_.end()) {
            return (int64_t) it->second;   // no append, no ref bump (pending state does)
        }
    }
    uint64_t p_off = 0;
    if (!I.append_record(CHUNK_REC_CHUNK, key, I.queued_gen_, data, len, p_off)) {
        // disk full guard: roll the batch back, caller retries on eviction
        I.abort_batch_locked();
        return -1;
    }
    I.batch_appended_ += I.rec_size(len);
    impl::chunk c;
    c.key   = key;
    c.offset = p_off;
    c.len    = (uint64_t) len;
    c.crc    = crc32c(data, len);
    c.refs   = 0;
    const uint64_t id = I.chunks_.size();
    I.chunks_.push_back(c);
    I.by_key_[key] = id;
    return (int64_t) id;
}

void prompt_cache_chunk_store::impl::abort_batch_locked() {
    if (queued_gen_ == 0) {
        return;
    }
    queued_gen_ = 0;
    batch_appended_ = 0;
    chunks_.resize(batch_chunks_);
    // drop any states queued this batch (tombs were never tracked in-memory)
    for (const auto & fp : batch_states_) {
        states_.erase(fp_str(fp));
        state_tokens_.erase(fp_str(fp));
    }
    batch_states_.clear();
    batch_tombs_.clear();
    truncate_to(batch_start_);
    // rebuild the key oracle from the surviving chunks
    by_key_.clear();
    for (size_t i = 0; i < chunks_.size(); i++) {
        by_key_[chunks_[i].key] = i;
    }
    recompute_refs();
}

void prompt_cache_chunk_store::impl::reset_log_locked() {
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
    chunks_.clear();
    states_.clear();
    state_tokens_.clear();
    by_key_.clear();
    committed_gen_  = 0;
    queued_gen_     = 0;
    batch_start_    = 0;
    batch_chunks_   = 0;
    batch_appended_ = 0;
    batch_states_.clear();
    batch_tombs_.clear();
    dci_valid_ = false;
    f_ = fopen(dcp_path_.c_str(), "w+b");
    if (!f_) {
        return;
    }
    if (!set_header_gen(0) || !sync_file()) {
        close_log_ = true;
    }
    std::error_code ec;
    std::filesystem::remove(dci_path_, ec);   // stale index: next open rescans
}

bool prompt_cache_chunk_store::put_state(const state_ref & st) {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0) {
        return false;
    }
    // every referenced chunk must exist and predate this state
    for (uint64_t id : st.chunks) {
        if (id >= I.chunks_.size()) {
            I.abort_batch_locked();
            return false;
        }
    }
    std::vector<uint8_t> payload(32 + 8 + 8 + 8 * st.chunks.size());
    memcpy(payload.data(), st.fp.data(), 32);
    cc_io::put_u64(payload.data() + 32, st.n_tokens);
    cc_io::put_u64(payload.data() + 40, st.chunks.size());
    for (size_t i = 0; i < st.chunks.size(); i++) {
        cc_io::put_u64(payload.data() + 48 + 8 * i, st.chunks[i]);
    }
    uint64_t p_off = 0;
    chunk_key key;
    if (!I.append_record(CHUNK_REC_STATE, key, I.queued_gen_, payload.data(), payload.size(), p_off)) {
        I.abort_batch_locked();
        return false;
    }
    I.batch_appended_ += I.rec_size(payload.size());
    for (uint64_t id : st.chunks) {
        I.chunks_[id].refs++;
    }
    I.states_[I.fp_str(st.fp)] = st.chunks;
    I.state_tokens_[I.fp_str(st.fp)] = st.n_tokens;
    I.batch_states_.push_back(st.fp);
    return true;
}

bool prompt_cache_chunk_store::commit() {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.queued_gen_ == 0) {
        return false;
    }
    if (!I.sync_file()) {
        I.abort_batch_locked();
        return false;
    }
    if (!I.set_header_gen(I.queued_gen_) || !I.sync_file()) {
        I.abort_batch_locked();
        return false;
    }
    I.committed_gen_ = I.queued_gen_;
    I.queued_gen_    = 0;
    I.batch_start_   = 0;
    I.batch_states_.clear();
    I.batch_tombs_.clear();
    I.dci_valid_ = false;   // cache.dci is now stale until write_index()
    return true;
}

void prompt_cache_chunk_store::abort_commit() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    I.abort_batch_locked();
}

bool prompt_cache_chunk_store::get_chunk(uint64_t id, std::vector<uint8_t> & out) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (id >= I.chunks_.size()) {
        return false;
    }
    const impl::chunk & c = I.chunks_[id];
    uint8_t rh[CHUNK_REC_HDR];
    if (!I.pread_at((int64_t) c.offset - CHUNK_REC_HDR, rh, CHUNK_REC_HDR)) {
        return false;
    }
    out.resize(c.len);
    if (c.len > 0 && !I.pread_at((int64_t) c.offset, out.data(), c.len)) {
        return false;
    }
    if (cc_io::get_u16(rh) != CHUNK_REC_CHUNK ||
        cc_io::get_u64(rh + 32) != c.len ||
        cc_io::get_u32(rh + 40) != (crc32c(rh, 40) ^ crc32c(out.data(), c.len))) {
        return false;
    }
    return true;
}

bool prompt_cache_chunk_store::get_state(const std::array<uint8_t, 32> & fp, state_ref & out) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    const std::string fp_s((const char *) fp.data(), 32);
    const auto it = I.states_.find(fp_s);
    if (it == I.states_.end()) {
        return false;
    }
    out.fp      = fp;
    out.chunks  = it->second;
    out.n_tokens = I.state_tokens_[fp_s];
    return true;
}

bool prompt_cache_chunk_store::has_state(const std::array<uint8_t, 32> & fp) const {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    return I.states_.find(std::string((const char *) fp.data(), 32)) != I.states_.end();
}

void prompt_cache_chunk_store::drop_state(const std::array<uint8_t, 32> & fp) {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    const std::string fp_s((const char *) fp.data(), 32);
    const auto it = I.states_.find(fp_s);
    if (it == I.states_.end()) {
        return;
    }
    if (!I.writable_) {
        return;
    }
    // tombstone must commit without a surrounding batch
    if (I.queued_gen_ != 0) {
        SRV_WRN("chunk store: drop_state ignored while a batch is open (gen %lu)", (unsigned long) I.queued_gen_);
        return;
    }
    std::vector<uint8_t> payload(32);
    memcpy(payload.data(), fp.data(), 32);
    chunk_key key;
    uint64_t p_off = 0;
    if (I.append_record(CHUNK_REC_TOMB, key, I.committed_gen_ + 1, payload.data(), payload.size(), p_off)) {
        if (!I.set_header_gen(I.committed_gen_ + 1) || !I.sync_file()) {
            return;
        }
        I.committed_gen_++;
        I.dci_valid_ = false;
        I.n_tombs_++;
        // unref and drop
        for (uint64_t id : it->second) {
            if (id < I.chunks_.size() && I.chunks_[id].refs > 0) {
                I.chunks_[id].refs--;
            }
        }
        I.states_.erase(fp_s);
        I.state_tokens_.erase(fp_s);
        I.recompute_refs();   // keep refs source-of-truth (tombstones may collapse)
    }
}

void prompt_cache_chunk_store::drop_evicted(const std::vector<std::array<uint8_t, 32>> & fps) {
    if (!p_ || fps.empty()) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return;
    }
    size_t n_dropped = 0;
    for (const auto & fp : fps) {
        const std::string fp_s((const char *) fp.data(), 32);
        if (I.states_.find(fp_s) == I.states_.end()) {
            continue;
        }
        // same tombstone as drop_state: durable so a crash-reload keeps the
        // eviction, and refcounts drop so compact can reclaim the chunks
        std::vector<uint8_t> payload(32);
        memcpy(payload.data(), fp.data(), 32);
        chunk_key key;
        uint64_t p_off = 0;
        if (I.append_record(CHUNK_REC_TOMB, key, I.committed_gen_ + 1, payload.data(), payload.size(), p_off)) {
            I.set_header_gen(I.committed_gen_ + 1);
            I.sync_file();
            I.committed_gen_++;
            I.dci_valid_ = false;
            I.n_tombs_++;
            for (uint64_t id : I.states_[fp_s]) {
                if (id < I.chunks_.size() && I.chunks_[id].refs > 0) {
                    I.chunks_[id].refs--;
                }
            }
            I.states_.erase(fp_s);
            I.state_tokens_.erase(fp_s);
            I.recompute_refs();
            n_dropped++;
        }
    }
    if (n_dropped > 0) {
        SRV_INF(" - disk prompt cache: chunk store dropped %zu evicted states\n", n_dropped);
    }
}

bool prompt_cache_chunk_store::should_compact(uint64_t disk_max_size) {
    if (!p_) {
        return false;
    }
    impl & I = *p_;
    if (!I.writable_ || I.queued_gen_ != 0) {
        return false;
    }

    uint64_t n_dead = 0, n_total = 0;
    {
        std::lock_guard<std::mutex> lk(I.io_mtx_);
        n_total = I.chunks_.size();
        for (const auto & c : I.chunks_) {
            if (c.refs == 0) {
                n_dead++;
            }
        }
    }

    if (n_total == 0) {
        return false;
    }

    const uint64_t n_live = n_total - n_dead;

    // Trigger 1: dead/live ratio > 35% (existing behavior, always checked)
    // also compact when n_live == 0 && n_dead > 0 (all chunks are dead)
    if (n_live == 0 && n_dead > 0) {
        return true;
    }
    if (n_live > 0 && n_dead * 100 / n_live > 35) {
        return true;
    }

    // Triggers 2 and 3 only fire when disk_max_size > 0 (vacuum + non-eviction path)
    if (disk_max_size == 0) {
        return false;
    }

    // Trigger 2: tomb/live ratio > 20% (persistent tombstones from evicted .dts)
    if (n_live > 0 && I.n_tombs_ > 0 && I.n_tombs_ * 100 / n_live > 20) {
        return true;
    }

    // Trigger 3: .dcp size > 50% of disk_max_size (eager, size-based)
    uint64_t dcp_size = 0;
    if (I.file_size_of(I.dcp_path_, dcp_size) && dcp_size > disk_max_size * 50 / 100) {
        return true;
    }

    return false;
}

void prompt_cache_chunk_store::compact_if_stale() {
    if (!p_) {
        return;
    }
    // compact_if_stale is called after drop_evicted in the eviction path.
    // it checks dead/live > 35% and tomb/live > 20% (size trigger is handled
    // separately in the non-eviction path of disk_run_state_job).
    if (!should_compact(0)) {
        return;
    }
    // compact() acquires io_mtx_ internally
    compact();
}

uint64_t prompt_cache_chunk_store::compact() {
    if (!p_) {
        return 0;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (!I.writable_ || I.queued_gen_ != 0) {
        return 0;
    }
    uint64_t before = 0;
    if (!I.file_size_of(I.dcp_path_, before)) {
        return 0;
    }
    uint64_t after = 0;
    {
        // rewrite only refcount>0 chunks + states, in id order -> new ids
        std::vector<int64_t> remap(I.chunks_.size(), -1);
        std::vector<const impl::chunk *> keep;
        for (size_t i = 0; i < I.chunks_.size(); i++) {
            if (I.chunks_[i].refs > 0) {
                remap[i] = (int64_t) keep.size();
                keep.push_back(&I.chunks_[i]);
            }
        }
        std::vector<uint64_t> new_offs(keep.size());
        const std::string tmp = I.dcp_path_ + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                return 0;
            }
            uint8_t hdr[24];
            cc_io::put_u32(hdr,      DCP_MAGIC);
            cc_io::put_u32(hdr + 4,  FORMAT_VER);
            cc_io::put_u64(hdr + 8,  I.base_lineage_);
            cc_io::put_u64(hdr + 16, I.committed_gen_ + 1);
            out.write((const char *) hdr, 24);

            uint64_t new_off = 24;
            for (size_t i = 0; i < keep.size(); i++) {
                const impl::chunk * c = keep[i];
                std::vector<uint8_t> payload(c->len);
                if (c->len > 0 && !I.pread_at((int64_t) c->offset, payload.data(), c->len)) {
                    return 0;
                }
                std::vector<uint8_t> h(CHUNK_REC_HDR);
                cc_io::put_u16(h.data(), CHUNK_REC_CHUNK);
                cc_io::put_u16(h.data() + 2, c->key.region);
                cc_io::put_u32(h.data() + 4, c->key.layer);
                cc_io::put_u32(h.data() + 8, c->key.cell_lo);
                cc_io::put_u32(h.data() + 12, c->key.cell_hi);
                cc_io::put_u64(h.data() + 16, c->key.lineage_epoch);
                cc_io::put_u64(h.data() + 24, I.committed_gen_ + 1);
                cc_io::put_u64(h.data() + 32, c->len);
                cc_io::put_u32(h.data() + 40, crc32c(h.data(), 40) ^ crc32c(payload.data(), c->len));
                cc_io::put_u32(h.data() + 44, 0);
                std::vector<uint8_t> rec(CHUNK_REC_HDR + ((c->len + 7) & ~(size_t) 7));
                memcpy(rec.data(), h.data(), CHUNK_REC_HDR);
                memcpy(rec.data() + CHUNK_REC_HDR, payload.data(), c->len);
                out.write((const char *) rec.data(), rec.size());
                new_offs[i] = new_off + CHUNK_REC_HDR;
                new_off   += I.rec_size(c->len);
            }
            // states, remapped
            for (const auto & it : I.states_) {
                std::vector<uint64_t> ids;
                ids.reserve(it.second.size());
                bool ok = true;
                for (uint64_t id : it.second) {
                    const int64_t nid = id < remap.size() ? remap[id] : -1;
                    if (nid < 0) {
                        ok = false;
                        break;
                    }
                    ids.push_back((uint64_t) nid);
                }
                if (!ok) {
                    return 0;   // state references a dropped chunk: corrupt, bail
                }
                std::array<uint8_t, 32> fp;
                memcpy(fp.data(), it.first.data(), 32);
                std::vector<uint8_t> payload(32 + 8 + 8 + 8 * ids.size());
                memcpy(payload.data(), fp.data(), 32);
                cc_io::put_u64(payload.data() + 32, I.state_tokens_[it.first]);
                cc_io::put_u64(payload.data() + 40, ids.size());
                for (size_t k = 0; k < ids.size(); k++) {
                    cc_io::put_u64(payload.data() + 48 + 8 * k, ids[k]);
                }
                std::vector<uint8_t> h(CHUNK_REC_HDR);
                h[0] = (uint8_t) CHUNK_REC_STATE;
                cc_io::put_u16(h.data() + 2, 0);
                cc_io::put_u32(h.data() + 8, 0);
                cc_io::put_u32(h.data() + 12, 0);
                cc_io::put_u64(h.data() + 16, 0);
                cc_io::put_u64(h.data() + 24, I.committed_gen_ + 1);
                cc_io::put_u64(h.data() + 32, payload.size());
                cc_io::put_u32(h.data() + 40, crc32c(h.data(), 40) ^ crc32c(payload.data(), payload.size()));
                cc_io::put_u32(h.data() + 44, 0);
                std::vector<uint8_t> rec(CHUNK_REC_HDR + ((payload.size() + 7) & ~(size_t) 7));
                memcpy(rec.data(), h.data(), CHUNK_REC_HDR);
                memcpy(rec.data() + CHUNK_REC_HDR, payload.data(), payload.size());
                out.write((const char *) rec.data(), rec.size());
            }
            out.flush();
            if (!out) {
                return 0;
            }
        }
        std::error_code ec;
        std::filesystem::rename(tmp, I.dcp_path_, ec);
        if (ec) {
            return 0;
        }
        if (I.f_) {
            fclose(I.f_);
        }
        I.f_ = fopen(I.dcp_path_.c_str(), "r+b");
        if (!I.f_) {
            return 0;
        }
        I.sync_file();

        // commit the compacted generation
        I.committed_gen_++;
        // rebuild the in-memory index with the new ids (keep no longer valid)
        std::vector<impl::chunk> new_chunks;
        new_chunks.reserve(keep.size());
        for (size_t i = 0; i < keep.size(); i++) {
            impl::chunk c = *keep[i];
            c.offset = new_offs[i];
            c.refs   = 0;
            new_chunks.push_back(c);
        }
        I.chunks_.swap(new_chunks);
        I.by_key_.clear();
        for (size_t i = 0; i < I.chunks_.size(); i++) {
            I.by_key_[I.chunks_[i].key] = i;
        }
        // states still carry pre-compact ids: remap them to the new layout
        for (auto & it : I.states_) {
            for (uint64_t & id : it.second) {
                id = (uint64_t) remap[id];
            }
        }
        I.recompute_refs();
        I.n_tombs_ = 0;   // tombstones are dropped by compaction

        if (!I.file_size_of(I.dcp_path_, after)) {
            after = before;
        }
        I.write_index_locked();
    }
    return before > after ? before - after : 0;
}

void prompt_cache_chunk_store::write_index() {
    if (!p_) {
        return;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    if (I.writable_) {
        I.write_index_locked();
    }
}

prompt_cache_chunk_store::stats prompt_cache_chunk_store::get_stats() const {
    stats s;
    if (!p_) {
        return s;
    }
    impl & I = *p_;
    std::lock_guard<std::mutex> lk(I.io_mtx_);
    s.n_chunks = I.chunks_.size();
    s.n_states = I.states_.size();
    for (const auto & c : I.chunks_) {
        if (c.refs == 0) {
            s.n_dead++;
        }
        s.payload_bytes += c.len;
    }
    uint64_t dcp = 0, dci = 0;
    I.file_size_of(I.dcp_path_, dcp);
    I.file_size_of(I.dci_path_, dci);
    s.store_bytes = dcp + dci;
    return s;
}

uint64_t prompt_cache_chunk_store::generation() const {
    return p_ ? p_->committed_gen_ : 0;
}

uint32_t prompt_cache_chunk_store::refcount(uint64_t id) const {
    if (!p_ || id >= p_->chunks_.size()) {
        return 0;
    }
    return p_->chunks_[id].refs;
}

std::vector<uint64_t> prompt_cache_chunk_store::cell_lineages(uint64_t base, const int32_t * tokens, size_t n_tokens) {
    // chain over 128-token cells: h_0 = sha256(base), h_i = sha256(h_prev || cell tokens)
    std::vector<uint64_t> out;
    out.reserve((n_tokens + 127) / 128);
    uint8_t h[32];
    {
        uint8_t b[8];
        cc_io::put_u64(b, base);
        server_prompt_cache_sha256_raw(b, 8, h);
    }
    for (size_t off = 0; off < n_tokens; off += 128) {
        const size_t n = std::min<size_t>(128, n_tokens - off);
        std::vector<uint8_t> buf(32 + 4 * n);
        memcpy(buf.data(), h, 32);
        for (size_t i = 0; i < n; i++) {
            cc_io::put_u32(buf.data() + 32 + 4 * i, (uint32_t) tokens[off + i]);
        }
        server_prompt_cache_sha256_raw(buf.data(), buf.size(), h);
        out.push_back(cc_io::get_u64(h));
    }
    return out;
}
