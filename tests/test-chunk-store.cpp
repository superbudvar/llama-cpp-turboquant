//  Functional test for the chunked append-only disk store (.dcp/.dci/.lock).
//
//  Runs the real prompt_cache_chunk_store codec round-trip paths against a
//  temp directory: dedup, refcounts, corruption handling on reopen (crc flip
//  and torn tail), compaction, the store_bytes budget metric, the always-append
//  mode (TURBO_DISK_CHUNK_STORE=0), the flock read-only fence, s9/s10 exercise
//  the llama_state_seq blob walker: canonical slice tile + byte-exact reassembly
//  and the exclusive path, s11 validates a full write-through + reopen
//  round-trip proving persisted chunks reassemble byte-exactly, s12 covers the
//  batch-evictee path (drop_evicted tombstones + refcount drop), and s13 the
//  dead/live ratio gate of compact_if_stale.
//
//  Scenario filter: pass "s1".."s13" as argv[1] to run that scenario alone.

#include "common.h"
#include "server-task.h"

#include "log.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

static void check(bool cond, const char * msg) {
    if (cond) {
        fprintf(stderr, "ok:   %s\n", msg);
    } else {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_failures++;
    }
}

static bool file_exists(const fs::path & path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

// Castagnoli crc32c, reflected; must match the store's own table bit for bit
static uint32_t crc32c_t(const uint8_t * data, size_t len) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int b = 0; b < 8; b++) {
                crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78u : 0);
            }
            table[i] = crc;
        }
        ready = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xff];
    }
    return crc;
}

static uint64_t rd_le64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) {
        v = (v << 8) | p[i];
    }
    return v;
}

static uint32_t rd_le32(const uint8_t * p) {
    return (uint32_t) (p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24));
}

// build one on-disk record byte-identical to the store's append_record bytes
static std::vector<uint8_t> build_record(uint16_t kind, uint16_t region, uint32_t layer,
                                         uint32_t cell_lo, uint32_t cell_hi, uint64_t lineage,
                                         uint64_t gen, const uint8_t * payload, size_t len) {
    std::vector<uint8_t> rec(48 + ((len + 7) & ~(size_t) 7));
    uint8_t * h = rec.data();
    cc_io::put_u16(h,       kind);
    cc_io::put_u16(h + 2,   region);
    cc_io::put_u32(h + 4,   layer);
    cc_io::put_u32(h + 8,   cell_lo);
    cc_io::put_u32(h + 12,  cell_hi);
    cc_io::put_u64(h + 16,  lineage);
    cc_io::put_u64(h + 24,  gen);
    cc_io::put_u64(h + 32,  len);
    cc_io::put_u32(h + 40,  crc32c_t(h, 40) ^ crc32c_t(payload, len));
    cc_io::put_u32(h + 44,  0);
    if (len > 0) {
        memcpy(rec.data() + 48, payload, len);
    }
    return rec;
}

static uint64_t content_tag(const std::vector<uint8_t> & v) {
    // mirror the write path: REGION_CKPT chunks key on a content tag of the
    // tile, not on the token chain (checkpoint payloads carry spec/beam state
    // that is not a pure function of the tokens). identical bytes must produce
    // the same tag, distinct bytes distinct tags
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : v) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

static prompt_cache_chunk_store::chunk_key key_of(uint16_t region, uint16_t layer,
                                                  uint32_t lo, uint32_t hi, uint64_t epoch) {
    prompt_cache_chunk_store::chunk_key k;
    k.region        = (uint8_t) region;
    k.layer         = layer;
    k.cell_lo       = lo;
    k.cell_hi       = hi;
    k.lineage_epoch = epoch;
    return k;
}

int main(int argc, char ** argv) {
    // optional filter: run only the named scenario
    std::string only = argc > 1 ? argv[1] : "";

    const fs::path tmp = fs::temp_directory_path() / ("tq-chunk-test-" + std::to_string((long) getpid()));
    std::error_code ec_tmp;
    fs::remove_all(tmp, ec_tmp);
    fs::create_directories(tmp, ec_tmp);

    constexpr uint64_t BASE = 0x11111111u;

    //
    //  scenario 1: codec round-trip + dedup oracle (no-dup)
    //
    if (only.empty() || only == "s1") {
        fprintf(stderr, "\n== s1: round-trip and dedup ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s1";
        check(cs.open(dir.string(), BASE), "s1: open creates the store");

        std::vector<uint8_t> cell(256);
        for (size_t i = 0; i < cell.size(); i++) {
            cell[i] = (uint8_t) (i * 7 + 3);
        }
        const auto k = key_of(0, 0, 0, 1, 7);

        int64_t id1 = -1, id2 = -1;
        check(cs.begin_commit(), "s1: begin_commit (batch 1)");
        id1 = cs.put_chunk(k, cell.data(), cell.size());
        check(id1 >= 0, "s1: put_chunk returns a chunk id");
        check(cs.commit(), "s1: commit (batch 1)");

        check(cs.begin_commit(), "s1: begin_commit (batch 2)");
        id2 = cs.put_chunk(k, cell.data(), cell.size());
        check(cs.commit(), "s1: commit (batch 2)");

        check(cs.wants_dedup(), "s1: dedup enabled by default");
        check(id1 == id2, "s1: repeated cell dedups to the same chunk id");
        check(cs.get_stats().n_chunks == 1, "s1: one record stored for two puts");
        check(cs.generation() == 2, "s1: generation advanced to committed batch");

        std::vector<uint8_t> back;
        check(cs.get_chunk((uint64_t) id1, back), "s1: get_chunk reads the chunk");
        check(back.size() == cell.size() && memcmp(back.data(), cell.data(), cell.size()) == 0,
              "s1: chunk bytes round-trip intact");

        std::array<uint8_t, 32> fp{};
        memcpy(fp.data(), "fp-A", 4);
        prompt_cache_chunk_store::state_ref st;
        st.fp = fp;
        st.n_tokens = 128;
        st.chunks   = { (uint64_t) id1 };
        check(cs.begin_commit(), "s1: begin_commit (state)");
        check(cs.put_state(st), "s1: put_state refs the chunk");
        check(cs.commit(), "s1: commit (state)");

        check(cs.has_state(fp), "s1: has_state finds the persisted state");
        prompt_cache_chunk_store::state_ref got;
        check(cs.get_state(fp, got), "s1: get_state returns the state");
        check(got.chunks.size() == 1 && got.chunks[0] == (uint64_t) id1, "s1: state chunk ids match");
        check(cs.refcount((uint64_t) id1) == 1, "s1: refcount is 1 after one state");

        cs.close();
    }

    //
    //  scenario 2: shared chunks across states, refcount down on drop
    //
    if (only.empty() || only == "s2") {
        fprintf(stderr, "\n== s2: refcount 1->2 and drop_state ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s2";
        check(cs.open(dir.string(), BASE), "s2: open creates the store");

        std::vector<uint8_t> a(128, 0x11), b(128, 0x22);
        const auto ka = key_of(0, 0, 0, 1, 5);
        const auto kb = key_of(0, 0, 1, 2, 5);

        check(cs.begin_commit(), "s2: begin_commit");
        const int64_t ida = cs.put_chunk(ka, a.data(), a.size());
        const int64_t idb = cs.put_chunk(kb, b.data(), b.size());
        check(ida == 0 && idb == 1, "s2: new chunk ids assigned");
        std::array<uint8_t, 32> fa{}, fb{};
        memcpy(fa.data(), "state-A", 7);
        memcpy(fb.data(), "state-B", 7);
        prompt_cache_chunk_store::state_ref sa, sb;
        sa.fp = fa; sa.n_tokens = 128; sa.chunks = { (uint64_t) ida };
        sb.fp = fb; sb.n_tokens = 256; sb.chunks = { (uint64_t) ida, (uint64_t) idb };
        check(cs.put_state(sa) && cs.put_state(sb), "s2: both states queued");
        check(cs.commit(), "s2: commit");

        check(cs.refcount((uint64_t) ida) == 2, "s2: shared prefix chunk refcount 2");
        check(cs.refcount((uint64_t) idb) == 1, "s2: tail chunk refcount 1");

        cs.drop_state(fb);
        check(cs.refcount((uint64_t) ida) == 1, "s2: refcount down to 1 after drop");
        check(cs.refcount((uint64_t) idb) == 0, "s2: tail chunk dead after drop");
        check(!cs.has_state(fb), "s2: dropped state gone");
        check(cs.has_state(fa), "s2: surviving state kept");
        check(cs.generation() == 2, "s2: tombstone bumped the generation");

        cs.close();
    }

    //
    //  scenario 3: corruption - a flipped payload byte is never returned
    //
    if (only.empty() || only == "s3") {
        fprintf(stderr, "\n== s3: crc-flip corruption ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s3";
        check(cs.open(dir.string(), BASE), "s3: open creates the store");

        std::vector<uint8_t> cell(256);
        for (size_t i = 0; i < cell.size(); i++) {
            cell[i] = (uint8_t) (i * 13 + 1);
        }
        const auto k = key_of(0, 0, 0, 1, 9);
        check(cs.begin_commit(), "s3: begin_commit");
        const int64_t id = cs.put_chunk(k, cell.data(), cell.size());
        std::array<uint8_t, 32> fp{};
        memcpy(fp.data(), "fp-c", 4);
        prompt_cache_chunk_store::state_ref st;
        st.fp = fp; st.n_tokens = 128; st.chunks = { (uint64_t) id };
        check(cs.put_state(st), "s3: put_state");
        check(cs.commit(), "s3: commit");
        cs.close();   // writes cache.dci

        // flip one payload byte (offset 24 hdr + 48 rec-hdr + 100 inside cell)
        const fs::path dcp = dir / "cache.dcp";
        std::error_code ec;
        std::vector<uint8_t> raw((size_t) fs::file_size(dcp, ec));
        {
            std::ifstream in(dcp, std::ios::binary);
            in.read((char *) raw.data(), (std::streamsize) raw.size());
        }
        raw[24 + 48 + 100] ^= 0xff;
        {
            std::ofstream out(dcp, std::ios::binary | std::ios::trunc);
            out.write((const char *) raw.data(), (std::streamsize) raw.size());
        }

        // fast path (.dci present): entry indexed, but the read oracle refuses
        prompt_cache_chunk_store cs2;
        check(cs2.open(dir.string(), BASE), "s3: reopen (fast path)");
        std::vector<uint8_t> bad;
        check(!cs2.get_chunk((uint64_t) id, bad), "s3: get_chunk refuses flipped bytes");
        cs2.close();

        // scan path (no .dci): the bad record is dropped at reopen
        fs::remove(dir / "cache.dci", ec);
        prompt_cache_chunk_store cs3;
        check(cs3.open(dir.string(), BASE), "s3: reopen (scan path)");
        check(cs3.get_stats().n_chunks == 0, "s3: corrupt record dropped by the scan");
        check(cs3.get_stats().n_states == 0, "s3: dependent state dropped too");
        cs3.close();
    }

    //
    //  scenario 4: torn tail - a record with a gen above the header is scrap
    //
    if (only.empty() || only == "s4") {
        fprintf(stderr, "\n== s4: torn-tail truncate ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s4";
        check(cs.open(dir.string(), BASE), "s4: open creates the store");

        std::vector<uint8_t> cell(256);
        for (size_t i = 0; i < cell.size(); i++) {
            cell[i] = (uint8_t) (i * 3 + 5);
        }
        const auto k = key_of(0, 0, 0, 1, 11);
        check(cs.begin_commit(), "s4: begin_commit");
        const int64_t id = cs.put_chunk(k, cell.data(), cell.size());
        check(cs.commit(), "s4: commit");
        cs.close();

        const fs::path dcp = dir / "cache.dcp";
        std::error_code ec;
        const uint64_t pre = (uint64_t) fs::file_size(dcp, ec);

        // append a fully valid record tagged with a gen one above the header:
        // an uncommitted batch that must vanish on the next open
        std::vector<uint8_t> hdr24(24);
        {
            std::ifstream in(dcp, std::ios::binary);
            in.read((char *) hdr24.data(), 24);
        }
        std::vector<uint8_t> ghost(64, 0xEE);
        const auto gkey = key_of(0, 0, 1, 2, 11);
        const std::vector<uint8_t> rec = build_record(
            (uint16_t) 1, gkey.region, gkey.layer, gkey.cell_lo, gkey.cell_hi,
            gkey.lineage_epoch, rd_le64(hdr24.data() + 16) + 1, ghost.data(), ghost.size());
        {
            std::ofstream out(dcp, std::ios::binary | std::ios::app);
            out.write((const char *) rec.data(), (std::streamsize) rec.size());
        }
        check((uint64_t) fs::file_size(dcp, ec) > pre, "s4: ghost record appended");

        fs::remove(dir / "cache.dci", ec);   // force the full scan
        prompt_cache_chunk_store cs2;
        check(cs2.open(dir.string(), BASE), "s4: reopen (scan path)");
        check(cs2.get_stats().n_chunks == 1, "s4: committed chunk kept");
        std::vector<uint8_t> back;
        check(cs2.get_chunk((uint64_t) id, back), "s4: committed chunk still readable");
check(back.size() == cell.size() && memcmp(back.data(), cell.data(), cell.size()) == 0,
              "s4: chunk bytes intact after truncate");
        check((uint64_t) fs::file_size(dcp, ec) == pre, "s4: torn tail truncated at open");
        cs2.close();
    }

    //
    //  scenario 5: compact reclaims dead bytes and keeps ids valid
    //
    if (only.empty() || only == "s5") {
        fprintf(stderr, "\n== s5: compact live bytes ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s5";
        check(cs.open(dir.string(), BASE), "s5: open creates the store");

        std::vector<uint8_t> a(256, 0x31), b(256, 0x32), c(256, 0x33);
        const auto ka = key_of(0, 0, 0, 1, 13);
        const auto kb = key_of(0, 0, 1, 2, 13);
        const auto kc = key_of(0, 0, 2, 3, 13);

        check(cs.begin_commit(), "s5: begin_commit (batch 1)");
        const int64_t ida = cs.put_chunk(ka, a.data(), a.size());
        const int64_t idb = cs.put_chunk(kb, b.data(), b.size());
        std::array<uint8_t, 32> fa{}, fb{};
        memcpy(fa.data(), "keep-A", 6);
        memcpy(fb.data(), "dead-B", 6);
        prompt_cache_chunk_store::state_ref sa, sb;
        sa.fp = fa; sa.n_tokens = 128; sa.chunks = { (uint64_t) ida, (uint64_t) idb };
        sb.fp = fb; sb.n_tokens = 128; sb.chunks = { (uint64_t) idb };
        check(cs.put_state(sa), "s5: put_state A");
        check(cs.commit(), "s5: commit (batch 1)");

        check(cs.begin_commit(), "s5: begin_commit (batch 2)");
        const int64_t idc = cs.put_chunk(kc, c.data(), c.size());
        sb.chunks = { (uint64_t) idc };
        check(cs.put_state(sb), "s5: put_state B (now only chunk c)");
        check(cs.commit(), "s5: commit (batch 2)");

        // A is now dead (chunks a and b unreferenced); B lives (chunk c)
        cs.drop_state(fa);
        check(cs.get_stats().n_dead == 2, "s5: two chunks dead before compact");

        const uint64_t freed = cs.compact();
        check(freed > 0, "s5: compact frees bytes");
        prompt_cache_chunk_store::stats st = cs.get_stats();
        check(st.n_chunks == 1, "s5: only the live chunk survives compact");
        check(st.n_dead == 0, "s5: no dead chunks after compact");
        check(st.n_states == 1, "s5: only the live state survives compact");

        prompt_cache_chunk_store::state_ref got;
        check(cs.get_state(fb, got), "s5: surviving state readable after compact");
        check(got.chunks.size() == 1, "s5: state ids remapped to the new layout");
        std::vector<uint8_t> back;
        check(cs.get_chunk(got.chunks[0], back), "s5: remapped chunk still readable");
        check(back.size() == c.size() && memcmp(back.data(), c.data(), c.size()) == 0,
              "s5: chunk c bytes intact after compact");

        cs.write_index();
        check(cs.get_stats().payload_bytes == c.size(), "s5: payload metric counts live bytes only");
        cs.close();
    }

    //
    //  scenario 6: budget metric - store_bytes == .dcp + .dci on disk
    //
    if (only.empty() || only == "s6") {
        fprintf(stderr, "\n== s6: budget metric ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s6";
        check(cs.open(dir.string(), BASE), "s6: open creates the store");

        std::vector<uint8_t> cell(256, 0x77);
        const auto k = key_of(0, 0, 0, 1, 17);
        check(cs.begin_commit(), "s6: begin_commit");
        check(cs.put_chunk(k, cell.data(), cell.size()) >= 0, "s6: put_chunk");
        check(cs.commit(), "s6: commit");
        cs.write_index();

        std::error_code ec;
        const uint64_t dcp = (uint64_t) fs::file_size(dir / "cache.dcp", ec);
        const uint64_t dci = (uint64_t) fs::file_size(dir / "cache.dci", ec);
        check(dcp > 0 && dci > 0, "s6: both files on disk");
        check(cs.get_stats().store_bytes == dcp + dci, "s6: store_bytes matches physical bytes");
        cs.close();
    }

    //
    //  scenario 7: TURBO_DISK_CHUNK_STORE=0 always appends
    //
    if (only.empty() || only == "s7") {
        fprintf(stderr, "\n== s7: always-append mode ==\n");

        setenv("TURBO_DISK_CHUNK_STORE", "0", 1);
        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s7";
        check(cs.open(dir.string(), BASE), "s7: open creates the store");
        check(!cs.wants_dedup(), "s7: dedup disabled by env");

        std::vector<uint8_t> cell(256, 0x5A);
        const auto k = key_of(0, 0, 0, 1, 19);
        check(cs.begin_commit(), "s7: begin_commit (batch 1)");
        const int64_t id1 = cs.put_chunk(k, cell.data(), cell.size());
        check(cs.commit(), "s7: commit (batch 1)");
        check(cs.begin_commit(), "s7: begin_commit (batch 2)");
        const int64_t id2 = cs.put_chunk(k, cell.data(), cell.size());
        check(cs.commit(), "s7: commit (batch 2)");

        check(id1 != id2, "s7: repeated cell appends again");
        check(cs.get_stats().n_chunks == 2, "s7: two records on disk");
        cs.close();
        unsetenv("TURBO_DISK_CHUNK_STORE");
    }

    //
    //  scenario 8: a second opener holds no lock - reads on, writes fence
    //
    if (only.empty() || only == "s8") {
        fprintf(stderr, "\n== s8: flock read-only fence ==\n");

        prompt_cache_chunk_store writer;
        const fs::path dir = tmp / "s8";
        check(writer.open(dir.string(), BASE), "s8: writer opens the store");

        std::vector<uint8_t> cell(128, 0x8A);
        const auto k = key_of(0, 0, 0, 1, 23);
        check(writer.begin_commit(), "s8: writer begin_commit");
        const int64_t id = writer.put_chunk(k, cell.data(), cell.size());
        std::array<uint8_t, 32> fp{};
        memcpy(fp.data(), "fp-8", 4);
        prompt_cache_chunk_store::state_ref st;
        st.fp = fp; st.n_tokens = 128; st.chunks = { (uint64_t) id };
        check(writer.put_state(st), "s8: writer put_state");
        check(writer.commit(), "s8: writer commit");

        // second opener in the same process: flock fails, read-only
        prompt_cache_chunk_store reader;
        check(reader.open(dir.string(), BASE), "s8: reader opens read-only");
        check(!reader.begin_commit(), "s8: read-only opener cannot begin a batch");
        std::vector<uint8_t> back;
        check(reader.get_chunk((uint64_t) id, back), "s8: read-only opener can read chunks");
        check(back.size() == cell.size() && memcmp(back.data(), cell.data(), cell.size()) == 0,
              "s8: reader sees the same bytes");
        check(reader.has_state(fp), "s8: read-only opener sees states");
        reader.drop_state(fp);
        check(reader.has_state(fp), "s8: read-only opener cannot drop states");
        check(reader.get_chunk((uint64_t) id, back), "s8: chunk intact after refused drop");

        writer.close();
        reader.close();
    }

    //
    //  scenario 9: blob walker - canonical slice map + exclusive path
    //
    if (only.empty() || only == "s9") {
        fprintf(stderr, "\n== s9: state blob walker ==\n");

        auto push_u32 = [](std::vector<uint8_t> & v, uint32_t x) {
            for (int i = 0; i < 4; i++) {
                v.push_back((uint8_t) (x >> (8 * i)));
            }
        };
        auto push_u64 = [](std::vector<uint8_t> & v, uint64_t x) {
            for (int i = 0; i < 8; i++) {
                v.push_back((uint8_t) (x >> (8 * i)));
            }
        };

        // real llama_state_seq layout: magic(4) + seq_id(4) + n_stream(4) +
        // cell_count(4) + meta rows immediately after (no stream tree). a
        // single-sequence contiguous write-through cell is [pos, 1, seq_id] at
        // stride 12, [pos, 1, ext(8), seq_id] at stride 20 (M-RoPE)
        auto build_blob = [&](uint32_t n_stream, uint32_t n_tokens, uint32_t v_trans,
                              uint32_t n_layer, uint32_t meta_ext) -> std::vector<uint8_t> {
            std::vector<uint8_t> v;
            push_u32(v, 0xaf143cd8);
            push_u32(v, 0);
            push_u32(v, n_stream);
            push_u32(v, n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                push_u32(v, i);
                push_u32(v, 1);
                if (meta_ext) {
                    push_u32(v, 0);
                    push_u32(v, 0);
                }
                push_u32(v, 0);
            }
            push_u32(v, v_trans);
            push_u32(v, n_layer);
            push_u32(v, 1);
            push_u64(v, 8);
            for (uint32_t i = 0; i < n_tokens * 8; i++) {
                v.push_back((uint8_t) i);
            }
            if (!v_trans) {
                push_u32(v, 1);
                push_u64(v, 8);
                for (uint32_t i = 0; i < n_tokens * 8; i++) {
                    v.push_back((uint8_t) (0x80 + i));
                }
            }
            return v;
        };

        const std::vector<uint8_t> ok = build_blob(1, 2, 0, 1, 0);
        check(ok.size() == 104, "s9: canonical blob is 104 bytes");

        std::vector<state_blob_slice> slices;
        const bool acc = slice_state_blob(ok.data(), ok.size(), 2, slices);
        check(acc, "s9: canonical blob accepted");
        if (acc) {
            check(slices.size() == 5, "s9: five slices (header, meta, table, K, V)");

            const size_t region_order[] = {
                (size_t) prompt_cache_chunk_store::REGION_HEADER,
                (size_t) prompt_cache_chunk_store::REGION_META_ROW,
                (size_t) prompt_cache_chunk_store::REGION_HEADER,
                (size_t) prompt_cache_chunk_store::REGION_K,
                (size_t) prompt_cache_chunk_store::REGION_V,
            };
            bool order_ok = slices.size() == 5;
            for (size_t i = 0; i < slices.size() && order_ok; i++) {
                order_ok = order_ok && (size_t) slices[i].region == region_order[i];
            }
            check(order_ok, "s9: slices in canonical blob order");

            check(slices[1].cell_lo == 0 && slices[1].cell_hi == 2, "s9: meta sheet covers both tokens");
            check(slices[3].layer == 0 && slices[3].cell_lo == 0 && slices[3].cell_hi == 2, "s9: K sheet covers the cell");
            check(slices[4].region == prompt_cache_chunk_store::REGION_V && slices[4].cell_hi == 2, "s9: V sheet present for non-MLA");

            // contiguous coverage: reassembly concatenates to the source bytes
            std::vector<uint8_t> rebuilt;
            for (const auto & sl : slices) {
                check(sl.byte_off + sl.byte_len <= ok.size(), "s9: slice in-range");
                rebuilt.insert(rebuilt.end(), ok.begin() + (ptrdiff_t) sl.byte_off,
                               ok.begin() + (ptrdiff_t) (sl.byte_off + sl.byte_len));
            }
            check(rebuilt == ok, "s9: reassembly is byte-exact");

            // the blob ends at the V block: last slice must be REGION_V, no tail
            check(slices.back().region == prompt_cache_chunk_store::REGION_V, "s9: no tail region when blob is exact");
        }

        // exclusive path: multi-stream
        const std::vector<uint8_t> ms = build_blob(2, 2, 0, 1, 0);
        check(!slice_state_blob(ms.data(), ms.size(), 2, slices), "s9: multi-stream rejected");

        // exclusive path: v_trans
        const std::vector<uint8_t> vt = build_blob(1, 2, 1, 0, 0);
        check(!slice_state_blob(vt.data(), vt.size(), 2, slices), "s9: v_trans rejected");

        // exclusive path: wrong token count
        check(!slice_state_blob(ok.data(), ok.size(), 3, slices), "s9: token mismatch rejected");
    }

    //
    //  scenario 10: blob walker - 20-byte meta cells + recurrent tail
    //
    if (only.empty() || only == "s10") {
        fprintf(stderr, "\n== s10: extended meta walk ==\n");

        auto push_u32 = [](std::vector<uint8_t> & v, uint32_t x) {
            for (int i = 0; i < 4; i++) {
                v.push_back((uint8_t) (x >> (8 * i)));
            }
        };
        auto push_u64 = [](std::vector<uint8_t> & v, uint64_t x) {
            for (int i = 0; i < 8; i++) {
                v.push_back((uint8_t) (x >> (8 * i)));
            }
        };

        // one stream, 2 tokens, M-RoPE meta (stride 20); two K layers; an
        // opaque tail that the best-effort V parse turns into REGION_TAIL
        // (MLA-shaped save)
        std::vector<uint8_t> v;
        push_u32(v, 0xaf143cd8);
        push_u32(v, 0);
        push_u32(v, 1);
        push_u32(v, 2);
        for (uint32_t i = 0; i < 2; i++) {
            push_u32(v, i);
            push_u32(v, 1);
            push_u32(v, 0);
            push_u32(v, 0);
            push_u32(v, 0);
        }
        push_u32(v, 0); // v_trans
        push_u32(v, 2); // n_layer
        push_u32(v, 1);
        push_u64(v, 16);
        for (uint32_t i = 0; i < 2 * 16; i++) {
            v.push_back((uint8_t) (0x30 + i));
        }
        push_u32(v, 1);
        push_u64(v, 16);
        for (uint32_t i = 0; i < 2 * 16; i++) {
            v.push_back((uint8_t) (0x50 + i));
        }
        for (int i = 0; i < 8; i++) {
            v.push_back((uint8_t) (0x90 + i)); // opaque tail
        }
        check(v.size() == 160, "s10: extended blob is 160 bytes");

        std::vector<state_blob_slice> slices;
        const bool acc = slice_state_blob(v.data(), v.size(), 2, slices);
        check(acc, "s10: extended blob accepted");
        if (acc) {
            check(slices.size() == 6, "s10: six slices (header, meta, table, K, K, tail)");

            check(slices[1].region == prompt_cache_chunk_store::REGION_META_ROW &&
                  slices[1].byte_len == 40, "s10: one 40-byte meta sheet for two 20-byte cells");
            check(slices[2].region == prompt_cache_chunk_store::REGION_HEADER &&
                  slices[2].byte_off == 56 && slices[2].byte_len == 8, "s10: kv table stays a 8-byte header chunk");
            check(slices[3].layer == 0 && slices[3].byte_off == 64 && slices[3].byte_len == 44, "s10: K layer 0 sheet");
            check(slices[4].layer == 1 && slices[4].byte_off == 108 && slices[4].byte_len == 44, "s10: K layer 1 sheet");
            check(slices[5].region == prompt_cache_chunk_store::REGION_TAIL &&
                  slices[5].byte_off == 152 && slices[5].byte_len == 8, "s10: tail sheet from the best-effort V parse");

            // contiguous coverage is byte-exact here too
            std::vector<uint8_t> rebuilt;
            for (const auto & sl : slices) {
                rebuilt.insert(rebuilt.end(), v.begin() + (ptrdiff_t) sl.byte_off,
                               v.begin() + (ptrdiff_t) (sl.byte_off + sl.byte_len));
            }
            check(rebuilt == v, "s10: reassembly is byte-exact");
        }
    }

    //
    //  scenario 11: full write-through + reopen round-trip. the walker slices a
    //  blob, put_chunk stores each slice, put_state commits the index, and a
    //  fresh open (simulating a cold boot) sees the same state; get_state +
    //  get_chunk reassembly must be byte-exact
    //
    if (only.empty() || only == "s11") {
        fprintf(stderr, "\n== s11: store round-trip reassembly ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s11";
        check(cs.open(dir.string(), BASE), "s11: open creates the store");

        // canonical single-stream blob matching s9's proven layout: header(16)
        // + meta(2*12) + v_trans/n_layer(8) + K(8+32) + V(8+32) = 104 bytes
        const uint32_t n_tokens = 2;
        auto push_u32_v = [](std::vector<uint8_t> & v, uint32_t x) {
            for (int i = 0; i < 4; i++) {
                v.push_back((uint8_t) (x >> (i * 8)));
            }
        };
        auto push_u64_v = [](std::vector<uint8_t> & v, uint64_t x) {
            for (int i = 0; i < 8; i++) {
                v.push_back((uint8_t) (x >> (i * 8)));
            }
        };
        std::vector<uint8_t> blob;
        push_u32_v(blob, 0xaf143cd8); // header magic
        push_u32_v(blob, 0);          // seq_id
        push_u32_v(blob, 1);          // n_stream
        push_u32_v(blob, n_tokens);   // cell_count
        for (uint32_t i = 0; i < n_tokens; i++) {
            push_u32_v(blob, i);   // kv_pos
            push_u32_v(blob, 1);   // n_seq_id
            push_u32_v(blob, 0);   // seq_id
        }
        push_u32_v(blob, 0); // v_trans
        push_u32_v(blob, 1); // n_layer
        // K block
        push_u32_v(blob, 1);
        push_u64_v(blob, 8);
        for (uint32_t i = 0; i < n_tokens * 8; i++) {
            blob.push_back((uint8_t) i);
        }
        // V block
        push_u32_v(blob, 1);
        push_u64_v(blob, 8);
        for (uint32_t i = 0; i < n_tokens * 8; i++) {
            blob.push_back((uint8_t) (0x80 + i));
        }
        check(blob.size() == 104, "s11: canonical blob is 104 bytes");

        // slice into chunks
        std::vector<state_blob_slice> slices;
        check(slice_state_blob(blob.data(), blob.size(), n_tokens, slices),
              "s11: slice_state_blob succeeds");

        // store each slice in the chunk store
        const uint64_t lineage = BASE;
        check(cs.begin_commit(), "s11: begin_commit (slices)");
        std::vector<int64_t> ids;
        for (const auto & sl : slices) {
            const auto k = key_of(sl.region, sl.layer, sl.cell_lo, sl.cell_hi, lineage);
            int64_t id = cs.put_chunk(k, blob.data() + (ptrdiff_t) sl.byte_off, sl.byte_len);
            check(id >= 0, "s11: put_chunk succeeded");
            ids.push_back(id);
        }
        check(cs.commit(), "s11: commit (slices)");

        // commit the state
        std::array<uint8_t, 32> fp;
        memcpy(fp.data(), "s11-fp", 6);
        prompt_cache_chunk_store::state_ref st;
        st.fp = fp;
        st.n_tokens = n_tokens;
        for (int64_t id : ids) {
            st.chunks.push_back((uint64_t) id);
        }
        check(cs.begin_commit(), "s11: begin_commit (state)");
        check(cs.put_state(st), "s11: put_state");
        check(cs.commit(), "s11: commit (state)");
        check(cs.has_state(fp), "s11: has_state finds the persisted state");

        // close and reopen (simulates a cold boot)
        cs.close();
        prompt_cache_chunk_store cs2;
        check(cs2.open(dir.string(), BASE), "s11: reopen on a fresh instance");
        check(cs2.has_state(fp), "s11: fresh instance sees the persisted state");

        // get_state + get_chunk reassembly: must be byte-exact
        prompt_cache_chunk_store::state_ref got;
        check(cs2.get_state(fp, got), "s11: get_state on fresh instance");
        check(got.n_tokens == n_tokens, "s11: n_tokens matches");

        std::vector<uint8_t> rebuilt;
        for (uint64_t id : got.chunks) {
            std::vector<uint8_t> cell;
            check(cs2.get_chunk(id, cell), "s11: get_chunk succeeded");
            rebuilt.insert(rebuilt.end(), cell.begin(), cell.end());
        }
        check(rebuilt == blob, "s11: reassembly after reopen is byte-exact");
    }

    //
    //  scenario 12: drop_evicted removes store states for pool-evicted .dts
    //  sidecars and leaves their chunks dead for compact to reclaim
    //
    if (only.empty() || only == "s12") {
        fprintf(stderr, "\n== s12: drop_evicted ==\n");

        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s12";
        check(cs.open(dir.string(), BASE), "s12: open creates the store");

        std::vector<uint8_t> a(256, 0x41), b(256, 0x42);
        const auto ka = key_of(0, 0, 0, 1, 29);
        const auto kb = key_of(0, 0, 1, 2, 29);
        check(cs.begin_commit(), "s12: begin_commit (batch 1)");
        const int64_t ida = cs.put_chunk(ka, a.data(), a.size());
        const int64_t idb = cs.put_chunk(kb, b.data(), b.size());
        std::array<uint8_t, 32> fa{}, fb{};
        memcpy(fa.data(), "evict-A", 7);
        memcpy(fb.data(), "keep-B", 6);
        prompt_cache_chunk_store::state_ref sa, sb;
        sa.fp = fa; sa.n_tokens = 128; sa.chunks = { (uint64_t) ida, (uint64_t) idb };
        sb.fp = fb; sb.n_tokens = 128; sb.chunks = { (uint64_t) idb };
        check(cs.put_state(sa), "s12: put_state A");
        check(cs.put_state(sb), "s12: put_state B");
        check(cs.commit(), "s12: commit (batch 1)");

        // A references chunk a and b; B references b only
        check(cs.refcount((uint64_t) ida) == 1, "s12: chunk a refcount 1");
        check(cs.refcount((uint64_t) idb) == 2, "s12: chunk b refcount 2");

        // evict A (simulates the pool pruning one .dts sidecar)
        cs.drop_evicted({ fa });
        check(!cs.has_state(fa), "s12: evicted state gone from memory");
        check(cs.has_state(fb), "s12: surviving state kept");
        check(cs.refcount((uint64_t) ida) == 0, "s12: chunk a now dead after eviction");
        check(cs.refcount((uint64_t) idb) == 1, "s12: chunk b refcount down to 1");
        prompt_cache_chunk_store::stats st = cs.get_stats();
        check(st.n_dead == 1, "s12: one dead chunk awaiting compact");

        // the tombstone must survive a reopen, mirroring drop_state durability
        std::vector<uint8_t> back;
        check(cs.get_chunk((uint64_t) idb, back), "s12: live chunk still readable");
        cs.close();
        prompt_cache_chunk_store cs2;
        check(cs2.open(dir.string(), BASE), "s12: reopen");
        check(!cs2.has_state(fa), "s12: eviction tombstone survives reopen");
        check(cs2.has_state(fb), "s12: surviving state survives reopen");
        check(cs2.refcount((uint64_t) ida) == 0, "s12: chunk a stays dead after reopen");
        cs2.close();
    }

    //
    //  scenario 13: compact_if_stale compacts only when dead/live ratio > 0.35
    //
    if (only.empty() || only == "s13") {
        fprintf(stderr, "\n== s13: compact_if_stale threshold ==\n");

        const auto seed = [&](prompt_cache_chunk_store & cs, uint64_t lineage, int n,
                              std::vector<std::array<uint8_t, 32>> & fps) {
            check(cs.begin_commit(), "s13: begin_commit");
            std::vector<int64_t> ids;
            for (int i = 0; i < n; i++) {
                std::vector<uint8_t> cell(128, (uint8_t) (0x10 + i));
                const auto k = key_of(0, 0, (uint32_t) i, (uint32_t) (i + 1), lineage);
                ids.push_back(cs.put_chunk(k, cell.data(), cell.size()));
            }
            // each fp references one distinct chunk
            for (int i = 0; i < n; i++) {
                std::array<uint8_t, 32> fp{};
                memcpy(fp.data(), "s13-fp", 6);
                fp[29] = (uint8_t) (lineage & 0xff);
                fp[30] = (uint8_t) i;
                prompt_cache_chunk_store::state_ref st;
                st.fp = fp; st.n_tokens = 64; st.chunks = { (uint64_t) ids[(size_t) i] };
                check(cs.put_state(st), "s13: put_state");
                fps.push_back(fp);
            }
            check(cs.commit(), "s13: commit");
        };

        // 4 chunks, 3 live states -> dead/live = 1/3 = 33% <= 35%: no compact
        {
            prompt_cache_chunk_store cs;
            const fs::path dir = tmp / "s13a";
            check(cs.open(dir.string(), BASE), "s13: open a");
            std::vector<std::array<uint8_t, 32>> fps;
            seed(cs, 37, 4, fps);
            cs.drop_evicted({ fps[3] });
            const uint64_t gen_before = cs.generation();
            cs.compact_if_stale();
            check(cs.generation() == gen_before, "s13: no compact below threshold (generation stable)");
            prompt_cache_chunk_store::stats st = cs.get_stats();
            check(st.n_chunks == 4, "s13: all chunks kept below threshold");
            cs.close();
        }

        // 4 chunks, 1 live state -> dead/live = 3/1 = 300% > 35%: compacts
        {
            prompt_cache_chunk_store cs;
            const fs::path dir = tmp / "s13b";
            check(cs.open(dir.string(), BASE), "s13: open b");
            std::vector<std::array<uint8_t, 32>> fps;
            seed(cs, 41, 4, fps);
            cs.drop_evicted({ fps[1], fps[2], fps[3] });
            cs.compact_if_stale();
            prompt_cache_chunk_store::stats st = cs.get_stats();
            check(st.n_chunks == 1, "s13: compact above threshold keeps one live chunk");
            check(st.n_dead == 0, "s13: no dead chunks after compact");
            cs.close();
        }
    }

    //  scenario 14: DTS v2 checkpoint-section codec. the sidecar writer (v2)
    //  appends the checkpoint section (mirror of the .dpc records) so a store
    //  restore can jump to its last checkpoint; v1 sidecars stay token-only.
    //  exercises the shared codec directly (disk_write/read_ckpt_section) the
    //  way load_from_disk_tier uses it
    //
    if (only.empty() || only == "s14") {
        fprintf(stderr, "\n== s14: dts v2 checkpoint codec ==\n");

        // two checkpoints with distinct metadata + non-trivial payloads
        std::list<common_prompt_checkpoint> src;
        for (int i = 0; i < 2; i++) {
            auto & c = src.emplace_back();
            c.n_tokens = 512 + i;
            c.id_task  = 10 + i;
            c.pos_min  = 100 + i;
            c.pos_max  = 900 + i;
            c.data_tgt.resize(64 + i);
            c.data_dft.resize(128 + i);
            c.data_spec.resize(0);
            for (size_t b = 0; b < c.data_tgt.size(); b++) {
                c.data_tgt[b] = (uint8_t) (0xa0 + i + b);
            }
            for (size_t b = 0; b < c.data_dft.size(); b++) {
                c.data_dft[b] = (uint8_t) (0x10 + i * b);
            }
        }

        // size oracle: record 0 carries 64 @ tgt + 128 @ dft payloads
        check(disk_ckpt_section_size(src, 1) == 44 + 64 + 128, "s14: size - one record with payloads");
        check(disk_ckpt_section_size(src, 0) == 0, "s14: size - zero records is zero");
        check(disk_ckpt_section_size(src, 2) == 44 + (64 + 128) + 44 + (65 + 129),
              "s14: size - both records with payloads");

        // tmpfile-based round trip: write the section, rewind, read it back
        FILE * f = tmpfile();
        check(f != nullptr, "s14: tmpfile opens");

        const uint32_t n_ckpt = (uint32_t) src.size();
        check(disk_write_ckpt_section(f, src, n_ckpt), "s14: write section");
        rewind(f);

        std::list<common_prompt_checkpoint> got;
        check(disk_read_ckpt_section(f, n_ckpt, got), "s14: read section");
        check(got.size() == src.size(), "s14: same checkpoint count");

        auto it_src = src.begin();
        auto it_got = got.begin();
        bool same = it_got != got.end();
        for (; same && it_src != src.end() && it_got != got.end(); ++it_src, ++it_got) {
            same = it_got->n_tokens == it_src->n_tokens &&
                   it_got->id_task  == it_src->id_task &&
                   it_got->pos_min  == it_src->pos_min &&
                   it_got->pos_max  == it_src->pos_max &&
                   it_got->data_tgt.size() == it_src->data_tgt.size() &&
                   it_got->data_dft.size() == it_src->data_dft.size() &&
                   it_got->data_spec.size() == it_src->data_spec.size() &&
                   memcmp(it_got->data_tgt.data(), it_src->data_tgt.data(), it_src->data_tgt.size()) == 0 &&
                   memcmp(it_got->data_dft.data(), it_src->data_dft.data(), it_src->data_dft.size()) == 0;
        }
        check(same, "s14: metadata and payload bytes round-trip intact");
        fclose(f);

        // n_ckpt over the cap is rejected before any read
        f = tmpfile();
        check(f != nullptr, "s14: tmpfile opens (overflow)");
        {
            std::list<common_prompt_checkpoint> got2;
            check(!disk_read_ckpt_section(f, DPC_MAX_CHECKPOINTS + 1, got2), "s14: n_ckpt over cap rejected");
            check(got2.empty(), "s14: no partial state on overflow");
        }
        fclose(f);

        // truncated section: valid first record, then a cut in the second
        f = tmpfile();
        check(f != nullptr, "s14: tmpfile opens (truncated)");
        {
            const uint32_t n_ckpt2 = (uint32_t) src.size();
            check(disk_write_ckpt_section(f, src, n_ckpt2), "s14: write section (truncated case)");
            rewind(f);

            // keep the first full record + 20 bytes of the second record, then cut
            const long cut     = (long) disk_ckpt_section_size(src, 1) + 20;
            FILE * g = tmpfile();
            std::vector<uint8_t> buf((size_t) cut);
            check(fread(buf.data(), 1, buf.size(), f) == buf.size(), "s14: copy prefix for truncation");
            fwrite(buf.data(), 1, buf.size(), g);
            rewind(g);

            std::list<common_prompt_checkpoint> got3;
            check(!disk_read_ckpt_section(g, n_ckpt2, got3), "s14: truncated section rejected");
            check(got3.empty(), "s14: no partial state on truncation");
            fclose(f);
            fclose(g);
        }

        // v1 semantics: a token-only sidecar has n_ckpt == 0 -> empty checkpoints
        f = tmpfile();
        check(f != nullptr, "s14: tmpfile opens (v1)");
        {
            std::list<common_prompt_checkpoint> got4;
            check(disk_write_ckpt_section(f, src, 0), "s14: v1 writes no section bytes");
            check(ftell(f) == 0, "s14: v1 section is empty");
            rewind(f);
            check(disk_read_ckpt_section(f, 0, got4), "s14: v1 read ok with n_ckpt 0");
            check(got4.empty(), "s14: v1 sidecar loads with empty checkpoints");
        }
        fclose(f);
    }

    //  scenario 15: v3 checkpoint chunks (REGION_CKPT). the store owns the
    //  checkpoint payloads as keyed chunks (stream/ordinal in layer, token span
    //  in cell_lo/cell_hi), so consecutive states dedup payloads by key; the
    //  sidecar keeps only 44-byte meta records. exercises chunk dedup, the meta
    //  codec round trip, the chunk-count oracle, and compact-renumber with the
    //  ckpt chunks referenced by a live state
    //
    if (only.empty() || only == "s15") {
        fprintf(stderr, "\n== s15: v3 ckpt chunks and meta codec ==\n");

        // two checkpoints produced by one data-dir beam: the second re-visits
        // the same token span, so it shares the first's tgt payload bytes
        std::list<common_prompt_checkpoint> src;
        for (int i = 0; i < 2; i++) {
            auto & c = src.emplace_back();
            c.n_tokens = 512 + i;
            c.id_task  = 10 + i;
            c.pos_min  = 100;
            c.pos_max  = 900;
            c.data_tgt.resize(128);
            c.data_dft.resize(64 + i);
            c.data_spec.resize(0);
            for (size_t b = 0; b < c.data_tgt.size(); b++) {
                c.data_tgt[b] = (uint8_t) (0xa0 + i + b);
            }
            for (size_t b = 0; b < c.data_dft.size(); b++) {
                c.data_dft[b] = (uint8_t) ((0x10 + i * b) & 0xff);
            }
        }
        // ckpt 2 reuses ckpt 1's tgt payload (the dedup case)
        src.back().data_tgt = src.front().data_tgt;

        // meta codec round trip: size oracle + bytes survive
        check(disk_ckpt_meta_size(0) == 0, "s15: meta size - zero records is zero");
        check(disk_ckpt_meta_size(2) == 88, "s15: meta size - 44 bytes per record");
        FILE * f = tmpfile();
        check(f != nullptr, "s15: tmpfile opens");
        const uint32_t n_ckpt = (uint32_t) src.size();
        check(disk_write_ckpt_meta(f, src, n_ckpt), "s15: write meta");
        check((uint64_t) ftell(f) == disk_ckpt_meta_size(n_ckpt), "s15: meta file is exactly 44*n_ckpt bytes");
        rewind(f);
        std::list<common_prompt_checkpoint> got;
        check(disk_read_ckpt_meta(f, n_ckpt, got), "s15: read meta");
        auto it_src = src.begin();
        auto it_got = got.begin();
        bool meta_same = it_got != got.end();
        for (; meta_same && it_src != src.end() && it_got != got.end(); ++it_src, ++it_got) {
            meta_same = it_got->n_tokens == it_src->n_tokens &&
                        it_got->id_task  == it_src->id_task &&
                        it_got->pos_min  == it_src->pos_min &&
                        it_got->pos_max  == it_src->pos_max &&
                        // the meta reader pre-sizes payloads to the header lengths
                        it_got->data_tgt.size() == it_src->data_tgt.size() &&
                        it_got->data_dft.size() == it_src->data_dft.size() &&
                        it_got->data_spec.size() == it_src->data_spec.size();
        }
        check(meta_same, "s15: meta round-trips lengths and metadata");
        fclose(f);

        // chunk-count oracle: streams tiled at MAX_REC_LEN, empty contributes 0
        check(disk_ckpt_chunk_count(0, src) == 0, "s15: chunk count - zero records is zero");
        {
            auto big = src.front();
            big.data_tgt.resize(prompt_cache_chunk_store::MAX_REC_LEN + 7);
            big.data_dft.resize(0);
            big.data_spec.resize(0);
            std::list<common_prompt_checkpoint> one = { big };
            check(disk_ckpt_chunk_count(1, one) == 2, "s15: chunk count - over-MAX_REC_LEN tgt tiles");
        }
        check(disk_ckpt_chunk_count(2, src) == 4,
              "s15: chunk count - 2 ckpts, empty spec stream, no tile boundary hit");

        // register the ckpt payloads as REGION_CKPT chunks in a real store,
        // then let state refs prove dedup + compact survive. the write path
        // keys each tile on a content tag (identical bytes -> same key), so a
        // repeated tgt payload must collapse while a differing dft must not
        prompt_cache_chunk_store cs;
        const fs::path dir = tmp / "s15";
        check(cs.open(dir.string(), BASE), "s15: open creates the store");

        struct ckpt_id { int64_t tgt, dft; };
        std::vector<ckpt_id> ids;
        check(cs.begin_commit(), "s15: begin_commit (ckpt chunks)");
        for (auto & c : src) {
            const uint16_t ord0 = 0;
            const auto kt = key_of(prompt_cache_chunk_store::REGION_CKPT, (uint16_t) (0 << 10 | ord0),
                                   (uint32_t) c.pos_min, (uint32_t) c.pos_max, content_tag(c.data_tgt));
            const auto kd = key_of(prompt_cache_chunk_store::REGION_CKPT, (uint16_t) (1 << 10 | ord0),
                                   (uint32_t) c.pos_min, (uint32_t) c.pos_max, content_tag(c.data_dft));
            ids.push_back({ cs.put_chunk(kt, c.data_tgt.data(), c.data_tgt.size()),
                            cs.put_chunk(kd, c.data_dft.data(), c.data_dft.size()) });
        }
        check(cs.commit(), "s15: commit (ckpt chunks)");
        // keys: tgt0==tgt1 (same payload -> same content tag), dft0 != dft1
        check(cs.get_stats().n_chunks == 3,
              "s15: identical tgt keys collapsed, differing dft keys did not");
        check(ids[1].tgt == ids[0].tgt, "s15: duplicated payload maps to the same chunk id");
        check(ids[1].dft != ids[0].dft, "s15: differing dft payload maps to a distinct chunk");
        check(cs.refcount((uint64_t) ids[1].tgt) == 0, "s15: no state yet, chunk unreferenced");

        // reopen to prove REGION_CKPT chunks are on-disk records like any other
        cs.close();
        check(cs.open(dir.string(), BASE), "s15: reopen creates the store");

        for (size_t i = 0; i < src.size(); i++) {
            auto it = src.begin();
            std::advance(it, (long) i);
            std::vector<uint8_t> back;
            check(cs.get_chunk((uint64_t) ids[i].tgt, back), "s15: tgt chunk read back");
            check(back.size() == it->data_tgt.size() &&
                          memcmp(back.data(), it->data_tgt.data(), it->data_tgt.size()) == 0,
                  "s15: tgt bytes intact across reopen");
            check(cs.get_chunk((uint64_t) ids[i].dft, back), "s15: dft chunk read back");
            check(back.size() == it->data_dft.size() &&
                          memcmp(back.data(), it->data_dft.data(), it->data_dft.size()) == 0,
                  "s15: dft bytes intact across reopen");
        }

        // two states like consecutive writes: both share the tgt chunk (the
        // C8 win), each has its own dft. drop the older state, then compact:
        // the shared ckpt chunk must survive the renumber and stay readable
        std::array<uint8_t, 32> fa{}, fb{};
        memcpy(fa.data(), "ckpt-state-A", 12);
        memcpy(fb.data(), "ckpt-state-B", 12);
        prompt_cache_chunk_store::state_ref sa, sb;
        sa.fp = fa; sa.n_tokens = 1024;
        sa.chunks = { (uint64_t) ids[0].tgt, (uint64_t) ids[0].dft };
        sb.fp = fb; sb.n_tokens = 2048;
        sb.chunks = { (uint64_t) ids[1].tgt, (uint64_t) ids[1].dft };
        check(cs.begin_commit(), "s15: begin_commit (states)");
        check(cs.put_state(sa) && cs.put_state(sb), "s15: both states queued");
        check(cs.commit(), "s15: commit (states)");
        check(cs.refcount((uint64_t) ids[0].tgt) == 2, "s15: shared tgt refcount 2 across states");

        cs.drop_state(fa);
        check(cs.refcount((uint64_t) ids[0].tgt) == 1, "s15: shared tgt held by B after dropping A");
        check(cs.get_stats().n_dead == 1, "s15: only A's dft is dead");

        const uint64_t freed = cs.compact();
        check(freed > 0, "s15: compact frees dead ckpt bytes");

        prompt_cache_chunk_store::state_ref got_state;
        check(cs.get_state(fb, got_state), "s15: ckpt-referencing state survives compact");
        check(got_state.chunks.size() == 2, "s15: state chunks remapped after compact");
        std::vector<uint8_t> back;
        check(cs.get_chunk(got_state.chunks[0], back), "s15: remapped ckpt chunk readable");
        check(!cs.get_state(fa, got_state), "s15: dropped state gone after compact");
        cs.close();
    }

    fs::remove_all(tmp, ec_tmp);

    if (g_failures) {
        fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }

    fprintf(stderr, "\nall ok\n");
    return 0;
}