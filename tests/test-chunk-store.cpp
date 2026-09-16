//  Functional test for the chunked append-only disk store (.dcp/.dci/.lock).
//
//  Runs the real prompt_cache_chunk_store codec round-trip paths against a
//  temp directory: dedup, refcounts, corruption handling on reopen (crc flip
//  and torn tail), compaction, the store_bytes budget metric, the always-append
//  mode (TURBO_DISK_CHUNK_STORE=0), and the flock read-only fence.
//
//  Scenario filter: pass "s1".."s8" as argv[1] to run that scenario alone.

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

    fs::remove_all(tmp, ec_tmp);

    if (g_failures) {
        fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }

    fprintf(stderr, "\nall ok\n");
    return 0;
}