//  Functional test for the shared disk prompt-cache pool.
//
//  Two server_prompt_cache instances (presets) live under ONE cache root. Each
//  preset dir holds the .dpc entries it wrote through plus its own .ccn spill.
//  A root-wide prune must:
//    - count .dpc across every preset dir (foreign .ccn never counted, never
//      evicted: it belongs to the owning worker's main thread)
//    - evict the oldest-mtime .dpc entries first
//    - bring the pool back under the shared disk_max_size budget
//
//  This is the mechanical evidence the CLI greps cannot provide: it runs the
//  actual prune path (server_prompt_cache::save_to_disk finishes with the
//  shared-pool prune) against a controlled directory tree.

#include "common.h"
#include "server-task.h"

#include "log.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

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

static void write_file(const fs::path & path, size_t size, const fs::file_time_type & mtime) {
    std::ofstream out(path, std::ios::binary);
    out.write("x", 1);
    out.seekp((std::streamoff) size - 1);
    out.write("x", 1);
    out.close();
    fs::last_write_time(path, mtime);
}

// bring the meta fingerprint in line so save_to_disk does not clear the dir
static void write_meta(const fs::path & dir, const std::string & fingerprint) {
    const fs::path meta_path = dir / "meta";
    std::ofstream out(meta_path, std::ios::binary);
    out.write(fingerprint.data(), (std::streamsize) fingerprint.size());
    out.close();
}

static bool file_exists(const fs::path & path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

// pool bytes: .dpc across all preset dirs (mirrors what prune_pool counts)
static uint64_t pool_dpc_bytes(const fs::path & root) {
    std::error_code ec;
    uint64_t total = 0;
    for (const auto & sd : fs::directory_iterator(root, ec)) {
        if (!sd.is_directory(ec)) {
            continue;
        }
        for (const auto & entry : fs::directory_iterator(sd.path(), ec)) {
            if (entry.path().extension() == ".dpc") {
                total += (uint64_t) fs::file_size(entry.path(), ec);
                if (ec) {
                    total -= (uint64_t) fs::file_size(entry.path(), ec);
                }
            }
        }
    }
    return total;
}

int main(int argc, char ** argv) {
    // optional filter: run only the named scenario
    std::string only = argc > 1 ? argv[1] : "";

    const fs::path tmp = fs::temp_directory_path() / ("tq-pool-test-" + std::to_string((long) getpid()));
    std::error_code ec_tmp;
    fs::remove_all(tmp, ec_tmp);
    fs::create_directories(tmp, ec_tmp);

    const fs::path root    = tmp / "cache-root";
    const fs::path dirA    = root / "preset-A";
    const fs::path dirB    = root / "preset-B";
    fs::create_directories(dirA);
    fs::create_directories(dirB);

    const std::string fingerprint = "test-fingerprint";

    // one MiB budget, shared by both instances (matches the router config)
    constexpr size_t budget_mib = 1;
    server_prompt_cache A(0, 0, budget_mib);
    server_prompt_cache B(0, 0, budget_mib);

    const fs::file_time_type now = fs::file_time_type::clock::now();

    //
    //  scenario 1: B's prune runs on a pool where A never hit its entry
    //
    if (only.empty() || only == "s1") {
        fprintf(stderr, "\n== scenario 1: cross-dir eviction, foreign .ccn untouched ==\n");

        // A wrote 0.6 MiB some time ago and spilled 0.4 MiB into its own .ccn
        const auto da = dirA / "a.dpc";
        const auto ca = dirA / "a.ccn";
        write_file(da, 600u * 1024, now - std::chrono::hours(2));
        write_file(ca, 400u * 1024, now - std::chrono::hours(2));

        // B wrote 0.6 MiB recently (never hit A's entry)
        write_file(dirB / "b.dpc", 600u * 1024, now - std::chrono::minutes(1));

        // fingerprints pre-seeded so neither cache treats the entries as stale
        write_meta(dirA, fingerprint);
        write_meta(dirB, fingerprint);

        // B's exit prune scans the whole root.
        B.prune_disk_budget(dirB.string());

        // (a) pool is back under budget (only B's .dpc remains: 0.6 MiB)
        check(pool_dpc_bytes(root) <= budget_mib * 1024 * 1024, "s1: used <= disk_max_size after prune");
        // (b) A's stale .dpc was evicted first (oldest mtime)
        check(!file_exists(da), "s1: A's stale .dpc evicted");
        check(file_exists(dirB / "b.dpc"), "s1: B's recent .dpc kept");
        // (c) A's foreign .ccn survived B's prune
        check(file_exists(ca), "s1: A's foreign .ccn untouched by B's prune");
    }

    //
    //  scenario 2: A's own exit prune may evict its own .ccn (ownership boundary)
    //
    if (only.empty() || only == "s2") {
        fprintf(stderr, "\n== scenario 2: owning worker may prune its own .ccn ==\n");

        fs::remove_all(root / "preset-A");
        fs::remove_all(root / "preset-B");
        fs::create_directories(dirA);
        fs::create_directories(dirB);

        // A spilled a large .ccn (0.8 MiB) and the pool is otherwise 0.4 MiB
        const auto da2 = dirA / "a.dpc";
        const auto ca2 = dirA / "a.ccn";
        write_file(da2, 200u * 1024, now - std::chrono::hours(1));
        write_file(ca2, 800u * 1024, now - std::chrono::hours(1));
        write_file(dirB / "b.dpc", 400u * 1024, now - std::chrono::minutes(5));

        write_meta(dirA, fingerprint);
        write_meta(dirB, fingerprint);

        // A's exit prune counts its OWN .ccn and evicts oldest first
        A.prune_disk_budget(dirA.string());

        // oldest in the pool is A's .ccn (same hour as a.dpc, but a.dpc was
        // written first in this scenario -> tie broken by file order is not
        // guaranteed; assert the invariant that matters: pool under budget and
        // A's untouched-by-foreign entry is gone while nothing of B's was touched)
        check(pool_dpc_bytes(root) <= budget_mib * 1024 * 1024, "s2: used <= disk_max_size after prune");
        check(file_exists(dirB / "b.dpc"), "s2: B's .dpc never touched by A's prune");
        check(!file_exists(ca2) || !file_exists(da2), "s2: A's own stale entry pruned");
    }

    //
    //  scenario 3: candidate selection must pick the LONGEST common prefix, not
    //  the first full-prefix entry.
    //
    //  regression driver: with an empty slot the old rule seeded f_keep_best=-1
    //  (any candidate wins), the first entry with f_keep=1.0 set f_keep_best=1.0,
    //  and `1.0 < f_keep_cur == 1.0` blocked every better successor -> the
    //  oldest snapshot was restored for a growing prompt. the correct pick is the
    //  candidate whose prefix reaches furthest into the new prompt.
    //
    if (only.empty() || only == "s3") {
        fprintf(stderr, "\n== scenario 3: selection picks the longest prefix ==\n");

        // growing conversation: 4 snapshots, each a full prefix of the next. all
        // have f_keep = 1.0, so the old strict-dominance rule froze on the first
        const std::vector<std::pair<size_t, size_t>> cands = {
            { 168100, 168100 },
            { 169171, 169171 },
            { 170474, 170474 },
            { 170874, 170874 }, // newest: longest prefix
        };

        // (a) empty slot, base lcp 0 -> longest prefix wins
        const size_t idx_a = server_prompt_cache::select_cache_candidate(cands, 0);
        check(idx_a == cands.size() - 1, "s3: longest full-prefix entry selected from empty slot");

        // (b) slot already holds a 168100-prefix -> must not regress, still picks newest
        const size_t idx_b = server_prompt_cache::select_cache_candidate(cands, 168100);
        check(idx_b == cands.size() - 1, "s3: longest prefix selected when slot already holds the oldest");

        // (c) slot already holds the newest prefix -> nothing qualifies
        check(
            server_prompt_cache::select_cache_candidate(cands, 170874) == SIZE_MAX,
            "s3: no candidate beats the slot's own newest prefix");

        // (d) don't-trash gate: entry extending the slot but keeping < 25% of its
        // own content is refused even when it is the best extension available
        const std::vector<std::pair<size_t, size_t>> trashing = {
            { 100000, 20000 }, // f_keep = 0.20 -> refused
            {  60000, 20000 }, // f_keep = 0.33 -> accepted
        };
        const size_t idx_d = server_prompt_cache::select_cache_candidate(trashing, 1000);
        check(idx_d == 1, "s3: don't-trash gate refuses the sub-0.25 entry");

        // (e) tie on lcp prefers the smaller state (preserves more of itself)
        const std::vector<std::pair<size_t, size_t>> tied = {
            { 80000, 40000 }, // same lcp as the next, larger state
            { 40000, 40000 },
        };
        const size_t idx_e = server_prompt_cache::select_cache_candidate(tied, 0);
        check(idx_e == 1, "s3: same-lcp tie broken toward the smaller state");
    }

    //
    //  scenario 4: chunk-store files (.dcp/.dci) count toward the pool budget
    //  but are never evicted by the prune; .dpc/.dts are the only candidates
    //
    if (only.empty() || only == "s4") {
        fprintf(stderr, "\n== scenario 4: chunk store files counted, never evicted ==\n");

        fs::remove_all(root / "preset-A");
        fs::remove_all(root / "preset-B");
        fs::create_directories(dirA);
        fs::create_directories(dirB);

        // pool: A has 0.4 MiB of .dpc plus the store writing 0.4 MiB .dcp and
        // 0.2 MiB .dci. budget is 1 MiB so the store files must be counted or
        // the prune would think the pool is only 0.6 MiB and evict nothing
        const auto da4 = dirA / "a.dpc";
        const auto dcp = dirA / "cache.dcp";
        const auto dci = dirA / "cache.dci";
        write_file(da4, 400u * 1024, now - std::chrono::hours(1));
        write_file(dcp, 400u * 1024, now - std::chrono::hours(1));
        write_file(dci, 200u * 1024, now - std::chrono::hours(1));
        write_file(dirB / "b.dpc", 200u * 1024, now - std::chrono::minutes(1));

        write_meta(dirA, fingerprint);
        write_meta(dirB, fingerprint);

        // The pool prune counts store bytes and evicts the oldest .dpc to fit.
        A.disk_prune_dpc(dirA.string(), nullptr);

        std::error_code ec4;
        uint64_t store_bytes = 0;
        if (file_exists(dcp)) store_bytes += (uint64_t) fs::file_size(dcp, ec4);
        if (file_exists(dci)) store_bytes += (uint64_t) fs::file_size(dci, ec4);
        check(pool_dpc_bytes(root) + store_bytes <= budget_mib * 1024 * 1024,
              "s4: pool under budget counting .dcp/.dci");
        check(file_exists(dcp) && file_exists(dci), "s4: store files never evicted by the prune");
        check(!file_exists(da4), "s4: A's .dpc evicted first (oldest candidate)");
        check(file_exists(dirB / "b.dpc"), "s4: B's recent .dpc kept");
    }

    //
    //  scenario 5 (oracle-8): back-compat + rollback. with the store disabled
    //  (TURBO_DISK_CHUNK_STORE=0) the boot index load prunes the store artifacts
    //  (.dcp/.dci/.dts/cache.lock) but leaves the legacy .dpc/.ccn tree intact
    //  and byte-identical - an older build reading the tree sees only what it
    //  knows, exactly as before the chunk store existed
    //
    if (only.empty() || only == "s5") {
        fprintf(stderr, "\n== scenario 5: rollback prune (store disabled) ==\n");

        const fs::path dirC = root / "preset-C";
        fs::remove_all(dirC, ec_tmp);
        fs::create_directories(dirC, ec_tmp);

        // legacy tree: one valid .dpc v3 header with zero tokens (32 bytes:
        // magic u32, version u32, n_tokens u32, len_main u64, len_drft u64,
        // n_ckpt u32), plus a .ccn with arbitrary checkpoint bytes
        uint32_t hdr5[8] = { 0x31435044, 3, 0, 0, 0, 0, 0, 0 };
        const fs::path dpc5 = dirC / "legacy.dpc";
        const fs::path ccn5 = dirC / "legacy.ccn";
        const std::string ccn5_body = "checkpoint-bytes";
        {
            std::ofstream out(dpc5, std::ios::binary);
            out.write((const char *) hdr5, (std::streamsize) sizeof(hdr5));
        }
        {
            std::ofstream out(ccn5, std::ios::binary);
            out.write(ccn5_body.data(), (std::streamsize) ccn5_body.size());
        }

        // store artifacts a store-enabled run left behind
        write_file(dirC / "cache.dcp", 100u * 1024, now - std::chrono::hours(1));
        write_file(dirC / "cache.dci",  25u * 1024, now - std::chrono::hours(1));
        write_file(dirC / "stale.dts",  10u * 1024, now - std::chrono::hours(1));
        write_file(dirC / "cache.lock", 1,          now - std::chrono::hours(1));

        write_meta(dirC, fingerprint);

        // snapshot the legacy tree before the rollback boot
        const fs::path dpc5_bak = tmp / "legacy.dpc.bak";
        const fs::path ccn5_bak = tmp / "legacy.ccn.bak";
        fs::copy_file(dpc5, dpc5_bak, fs::copy_options::overwrite_existing, ec_tmp);
        fs::copy_file(ccn5, ccn5_bak, fs::copy_options::overwrite_existing, ec_tmp);

        // disable the store for this boot; restore the env slot afterwards
        char * prev = getenv("TURBO_DISK_CHUNK_STORE");
        const std::string prev_str = prev ? prev : "";
        setenv("TURBO_DISK_CHUNK_STORE", "0", 1);

        const size_t n_idx = A.load_disk_index(dirC.string(), fingerprint);
        check(n_idx == 0, "s5: legacy .dpc is not indexed by the DTS-only loader");
        check(!file_exists(dirC / "cache.dcp"),     "s5: cache.dcp pruned (rollback)");
        check(!file_exists(dirC / "cache.dci"),     "s5: cache.dci pruned (rollback)");
        check(!file_exists(dirC / "stale.dts"),     "s5: .dts sidecar pruned (rollback)");
        check(!file_exists(dirC / "cache.lock"),    "s5: cache.lock pruned (rollback)");
        check(file_exists(dpc5), "s5: legacy .dpc kept");
        check(file_exists(ccn5), "s5: legacy .ccn kept");

        // byte-for-byte identical legacy files after the prune
        {
            std::ifstream a(dpc5_bak, std::ios::binary);
            std::ifstream b(dpc5, std::ios::binary);
            check(std::equal(std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>(),
                             std::istreambuf_iterator<char>(b)),
                  "s5: legacy .dpc bytes unchanged");
        }
        {
            std::ifstream a(ccn5_bak, std::ios::binary);
            std::ifstream b(ccn5, std::ios::binary);
            check(std::equal(std::istreambuf_iterator<char>(a), std::istreambuf_iterator<char>(),
                             std::istreambuf_iterator<char>(b)),
                  "s5: legacy .ccn bytes unchanged");
        }
        fs::remove(dpc5_bak, ec_tmp);
        fs::remove(ccn5_bak, ec_tmp);

        if (prev) {
            setenv("TURBO_DISK_CHUNK_STORE", prev_str.c_str(), 1);
        } else {
            unsetenv("TURBO_DISK_CHUNK_STORE");
        }
    }

    //
    //  scenario 6: chunk-store dedup - shared prefix + divergent tail
    //
    //  Two states sharing a long token prefix are written to the same store.
    //  The store must dedup the shared cells, so physical size < sum of two
    //  full blobs (the .dpc baseline). restore of both states must be
    //  byte-exact.
    //
    if (only.empty() || only == "s6") {
        fprintf(stderr, "\n== scenario 6: chunk-store dedup measurement ==\n");

        const fs::path store_dir = root / "preset-D";
        fs::remove_all(store_dir, ec_tmp);
        fs::create_directories(store_dir, ec_tmp);
        write_meta(store_dir, fingerprint);

        constexpr uint64_t BASE = 42;

        prompt_cache_chunk_store store;
        setenv("TURBO_DISK_CHUNK_STORE", "1", 1);
        check(store.open(store_dir.string(), BASE), "s6: store opens");
        check(store.is_open(), "s6: store is_open");

        // state A: 2048 tokens across 2 layers (K+V)
        // state B: 2560 tokens, sharing A's first 2048 (80% prefix)
        constexpr uint32_t TOKENS_A = 2048;
        constexpr uint32_t TOKENS_B = 2560;
        constexpr uint32_t N_LAYER  = 2;
        constexpr uint32_t CELL     = 128;
        constexpr uint32_t CELLS_A  = TOKENS_A / CELL;  // 16
        constexpr uint32_t CELLS_B  = (TOKENS_B + CELL - 1) / CELL; // 20
        constexpr uint32_t N_CELLS_SHARED = CELLS_A;    // first 2048 tokens shared
        constexpr size_t   CELL_PAYLOAD   = 512;        // 512 bytes per cell sheet

        // synthetic token ids: A = 0..2047, B = 0..2559 (B extends A)
        std::vector<int32_t> tokens_a(TOKENS_A);
        std::vector<int32_t> tokens_b(TOKENS_B);
        for (uint32_t i = 0; i < TOKENS_A; i++) {
            tokens_a[i] = (int32_t) i;
            tokens_b[i] = (int32_t) i;
        }
        for (uint32_t i = TOKENS_A; i < TOKENS_B; i++) {
            tokens_b[i] = (int32_t) i;
        }

        const auto lineages_a = prompt_cache_chunk_store::cell_lineages(BASE, tokens_a.data(), tokens_a.size());
        const auto lineages_b = prompt_cache_chunk_store::cell_lineages(BASE, tokens_b.data(), tokens_b.size());

        check(lineages_a.size() == CELLS_A, "s6: lineages_a size");
        check(lineages_b.size() == CELLS_B, "s6: lineages_b size");

        // verify shared prefix produces equal lineage epochs
        bool lineage_match = true;
        for (uint32_t c = 0; c < N_CELLS_SHARED; c++) {
            if (lineages_a[c] != lineages_b[c]) {
                lineage_match = false;
                break;
            }
        }
        check(lineage_match, "s6: shared prefix -> equal lineage epochs");

        // encode two synthetic payloads (A shorter, B longer)
        auto make_blob = [&](uint32_t n_tokens) -> std::vector<uint8_t> {
            const uint32_t n_cells = (n_tokens + CELL - 1) / CELL;
            size_t sz = 0;
            // header: 1 chunk
            sz += 64;
            // meta rows: n_cells
            sz += (size_t) n_cells * CELL_PAYLOAD;
            // K/V cell sheets: N_LAYER rows, alternating region by layer
            sz += (size_t) N_LAYER * n_cells * CELL_PAYLOAD;
            // tail: 1 chunk
            sz += 64;
            std::vector<uint8_t> blob(sz, 0xAB);
            // distinguish regions so dedup-vs-not is visible in the byte stream
            for (size_t i = 64 + (size_t) n_cells * CELL_PAYLOAD; i + CELL_PAYLOAD <= sz; i += CELL_PAYLOAD) {
                blob[i] = 0xCD;
            }
            return blob;
        };

        const auto blob_a = make_blob(TOKENS_A);
        const auto blob_b = make_blob(TOKENS_B);
        check(blob_a.size() < blob_b.size(), "s6: blob_a < blob_b");

        // put_state helper: writes all cell sheets + header + tail
        auto put_state = [&](const std::vector<int32_t> & tokens,
                             const std::vector<uint64_t> & lineages,
                             const std::vector<uint8_t> & blob,
                             const std::string & label,
                             std::array<uint8_t, 32> & fp_out,
                             uint64_t & n_cells_out) -> bool {
            const uint32_t n_tok  = (uint32_t) tokens.size();
            const uint32_t n_cells = (n_tok + CELL - 1) / CELL;
            const uint64_t last_epoch = lineages.empty() ? BASE : lineages.back();

            auto epoch_of = [&](uint32_t cell_idx) -> uint64_t {
                return cell_idx < lineages.size() ? lineages[cell_idx] : last_epoch;
            };

            if (!store.begin_commit()) {
                return false;
            }
            std::vector<uint64_t> ids;

            // header chunk
            {
                prompt_cache_chunk_store::chunk_key key;
                key.region = prompt_cache_chunk_store::REGION_HEADER;
                key.layer  = 0;
                key.cell_lo = 0;
                key.cell_hi = n_tok;
                key.lineage_epoch = last_epoch;
                int64_t id = store.put_chunk(key, blob.data(), 64);
                if (id < 0) { store.abort_commit(); return false; }
                ids.push_back((uint64_t) id);
            }

            // meta rows: one per cell
            const size_t meta_off = 64;
            for (uint32_t c = 0; c < n_cells; c++) {
                prompt_cache_chunk_store::chunk_key key;
                key.region = prompt_cache_chunk_store::REGION_META_ROW;
                key.layer  = 0;
                key.cell_lo = c * CELL;
                key.cell_hi = std::min(n_tok, (c + 1) * CELL);
                key.lineage_epoch = epoch_of(c);
                size_t off = meta_off + (size_t) c * CELL_PAYLOAD;
                int64_t id = store.put_chunk(key, blob.data() + off, CELL_PAYLOAD);
                if (id < 0) { store.abort_commit(); return false; }
                ids.push_back((uint64_t) id);
            }

            // K and V cell sheets per layer
            const size_t kv_start = meta_off + (size_t) n_cells * CELL_PAYLOAD;
            for (uint32_t l = 0; l < N_LAYER; l++) {
                for (uint32_t c = 0; c < n_cells; c++) {
                    prompt_cache_chunk_store::chunk_key key;
                    key.region = (l % 2 == 0) ? prompt_cache_chunk_store::REGION_K
                                              : prompt_cache_chunk_store::REGION_V;
                    key.layer  = l;
                    key.cell_lo = c * CELL;
                    key.cell_hi = std::min(n_tok, (c + 1) * CELL);
                    key.lineage_epoch = epoch_of(c);
                    size_t off = kv_start + (size_t) l * (n_cells * CELL_PAYLOAD) + (size_t) c * CELL_PAYLOAD;
                    int64_t id = store.put_chunk(key, blob.data() + off, CELL_PAYLOAD);
                    if (id < 0) { store.abort_commit(); return false; }
                    ids.push_back((uint64_t) id);
                }
            }

            // tail chunk
            {
                prompt_cache_chunk_store::chunk_key key;
                key.region = prompt_cache_chunk_store::REGION_TAIL;
                key.layer  = 0;
                key.cell_lo = 0;
                key.cell_hi = n_tok;
                key.lineage_epoch = last_epoch;
                size_t tail_off = blob.size() - 64;
                int64_t id = store.put_chunk(key, blob.data() + tail_off, 64);
                if (id < 0) { store.abort_commit(); return false; }
                ids.push_back((uint64_t) id);
            }

            prompt_cache_chunk_store::state_ref st;
            for (size_t i = 0; i < 32; i++) {
                st.fp[i] = (uint8_t) (label[i % label.size()] + i);
            }
            st.n_tokens = tokens.size();
            st.chunks   = std::move(ids);
            if (!store.put_state(st) || !store.commit()) {
                store.abort_commit();
                return false;
            }
            fp_out = st.fp;
            n_cells_out = n_cells;
            fprintf(stderr, "ok:   s6: wrote state %s (%u tokens, %u cells)\n", label.c_str(), n_tok, n_cells);
            return true;
        };

        std::array<uint8_t, 32> fp_a{}, fp_b{};
        uint64_t cells_a = 0, cells_b = 0;
        check(put_state(tokens_a, lineages_a, blob_a, "A", fp_a, cells_a), "s6: put_state A");
        check(put_state(tokens_b, lineages_b, blob_b, "B", fp_b, cells_b), "s6: put_state B");
        check(fp_a != fp_b, "s6: fingerprints differ");

        const auto st = store.get_stats();
        fprintf(stderr, "ok:   s6: store stats: %zu chunks, %zu dead, %zu states, %lu payload, %lu disk\n",
                st.n_chunks, st.n_dead, st.n_states, st.payload_bytes, st.store_bytes);

        // --- dedup measurement ---
        // duplicated baseline: each state stored as a full .dpc-sized blob
        const size_t dup_baseline = blob_a.size() + blob_b.size();
        fprintf(stderr, "ok:   s6: duplicated baseline (sum of full blobs): %zu bytes\n", dup_baseline);
        fprintf(stderr, "ok:   s6: store payload (unique chunks):         %lu bytes\n", st.payload_bytes);
        fprintf(stderr, "ok:   s6: store disk (cache.dcp + cache.dci):   %lu bytes\n", st.store_bytes);
        const double ratio = (st.payload_bytes < dup_baseline)
            ? 1.0 - (double) st.payload_bytes / (double) dup_baseline
            : 0.0;
        fprintf(stderr, "ok:   s6: dedup saving: %.1f%% (%.2fx compression vs duplicated)\n",
                ratio * 100.0, (double) dup_baseline / (double) st.payload_bytes);

        check(st.payload_bytes < dup_baseline, "s6: store payload < duplicated baseline");

        // --- byte-exact restore ---
        {
            prompt_cache_chunk_store::state_ref ra;
            check(store.get_state(fp_a, ra), "s6: get_state A");
            check(ra.n_tokens == TOKENS_A, "s6: A n_tokens");
            std::vector<uint8_t> reassembled;
            for (uint64_t cid : ra.chunks) {
                std::vector<uint8_t> chunk;
                check(store.get_chunk(cid, chunk), "s6: get_chunk for A");
                reassembled.insert(reassembled.end(), chunk.begin(), chunk.end());
            }
            check(reassembled == blob_a, "s6: A byte-exact restore");
        }
        {
            prompt_cache_chunk_store::state_ref rb;
            check(store.get_state(fp_b, rb), "s6: get_state B");
            check(rb.n_tokens == TOKENS_B, "s6: B n_tokens");
            std::vector<uint8_t> reassembled;
            for (uint64_t cid : rb.chunks) {
                std::vector<uint8_t> chunk;
                check(store.get_chunk(cid, chunk), "s6: get_chunk for B");
                reassembled.insert(reassembled.end(), chunk.begin(), chunk.end());
            }
            check(reassembled == blob_b, "s6: B byte-exact restore");
        }

        // verify dedup: B reuses A's shared chunk ids (the id returned by put_chunk
        // must match for the same key)
        {
            prompt_cache_chunk_store::state_ref rb;
            check(store.get_state(fp_b, rb), "s6: get_state B (dedup check)");
            // B has N_CELLS_SHARED*3+2 cells: header + 1 meta + 2 layers per cell + tail
            // of those, the shared-prefix cells should have the same ids as A's
            prompt_cache_chunk_store::state_ref ra;
            check(store.get_state(fp_a, ra), "s6: get_state A (dedup check)");
            // at least the shared cells must be the same ids (dedup)
            size_t shared_count = 0;
            for (uint64_t id_a : ra.chunks) {
                for (uint64_t id_b : rb.chunks) {
                    if (id_a == id_b) {
                        shared_count++;
                        break;
                    }
                }
            }
            fprintf(stderr, "ok:   s6: B reuses %zu chunk ids from A (of %zu + %zu total)\n",
                    shared_count, ra.chunks.size(), rb.chunks.size());
            check(shared_count > 0, "s6: B deduplicates at least some chunks from A");
        }

        store.close();
    }

    //
    //  scenario 7: free-space eviction below budget (evict-first guard).
    //
    //  A volume that is full (more precisely: min_free larger than what the
    //  volume can ever hold) must evict the oldest .dpc/.dts across the shared
    //  pool even when the pool is UNDER its byte budget. the 1<<60 min_free
    //  target is unreachable, so the free-space pass runs to exhaustion and the
    //  test observes which files it took. store files (.dcp/.dci) and every
    //  .ccn must survive: the store owns the former, the main thread owns the
    //  latter.
    //
    if (only.empty() || only == "s7") {
        fprintf(stderr, "\n== scenario 7: free-space eviction below budget ==\n");

        fs::remove_all(root / "preset-A");
        fs::remove_all(root / "preset-B");
        fs::create_directories(dirA);
        fs::create_directories(dirB);

        // header constants mirror the current DTS v4 parser in server-task.cpp
        constexpr uint32_t DTS_MAGIC   = 0x31535444; // "DTS1"
        constexpr uint32_t DTS_VERSION = 4;

        // pool is far under the 1 MiB budget: ~64 KiB of cache files total
        const auto a_dpc = dirA / "oldest.dpc";
        const auto a_dts = dirA / "state.dts";
        const auto b_dpc = dirB / "b.dpc";
        const auto a_dcp = dirA / "cache.dcp";
        const auto a_dci = dirA / "cache.dci";
        const auto a_ccn = dirA / "own.ccn";
        const auto b_ccn = dirB / "foreign.ccn";

        write_file(a_dpc, 32u * 1024, now - std::chrono::hours(2));
        write_file(b_dpc, 16u * 1024, now - std::chrono::hours(1));
        write_file(a_dcp, 8u * 1024,  now - std::chrono::hours(1));
        write_file(a_dci, 4u * 1024,  now - std::chrono::hours(1));
        write_file(a_ccn, 2u * 1024,  now - std::chrono::hours(1));
        write_file(b_ccn, 2u * 1024,  now - std::chrono::hours(1));

        // a valid .dts v4 sidecar: header, tokens, checkpoint metadata.
        {
            constexpr uint32_t N_TOK = 8;
            uint32_t hdr[3] = { DTS_MAGIC, DTS_VERSION, N_TOK };
            std::ofstream out(a_dts, std::ios::binary);
            out.write((const char *) hdr, sizeof(hdr));
            uint64_t len[2] = { 0, 0 };
            uint32_t n_ckpt = 0;
            out.write((const char *) len, sizeof(len));
            out.write((const char *) &n_ckpt, sizeof(n_ckpt));
            uint64_t last_access = 0;
            out.write((const char *) &last_access, sizeof(last_access));
            for (uint32_t i = 0; i < N_TOK; i++) {
                int32_t t = (int32_t) (100 + i);
                out.write((const char *) &t, sizeof(t));
            }
        }

        write_meta(dirA, fingerprint);
        write_meta(dirB, fingerprint);

        // unreachable free-space target -> free-space pass evicts to exhaustion
        std::vector<std::array<uint8_t, 32>> evicted;
        B.disk_prune_dpc(dirB.string(), &evicted, 1ULL << 60);

        check(!file_exists(a_dpc), "s7: oldest cross-preset .dpc evicted for free space");
        check(!file_exists(a_dts), "s7: .dts sidecar evicted for free space");
        check(!file_exists(b_dpc), "s7: own .dpc evicted for free space");
        check(file_exists(a_dcp) && file_exists(a_dci), "s7: store files never evicted by free space");
        check(file_exists(a_ccn), "s7: own .ccn never evicted by free space");
        check(file_exists(b_ccn), "s7: foreign .ccn never evicted by free space");
        check(evicted.size() == 1, "s7: evicted carries the .dts fingerprint");
    }

    //
    //  scenario 8 (B1): knob B loop evicts ALL own .dts, not just one.
    //  before the fix: the loop broke after one eviction, leaving the store
    //  over cap with most .dts intact. now it evicts until own_foot <= cap.
    //
    if (only.empty() || only == "s8") {
        fprintf(stderr, "\n== scenario 8: knob B evicts ALL own .dts ==\n");

        fs::remove_all(root / "preset-A");
        fs::remove_all(root / "preset-B");
        fs::create_directories(dirA);
        fs::create_directories(dirB);

        // disk_max_store_size = 60% of 1 MiB pool = 0.6 MiB cap = 614400 bytes.
        // create a store of 680 KiB (over cap) plus 3 .dts files (~64B each).
        // after evicting one .dts (64B): 680 - 64 = 679.9 KiB > 614.4 KiB cap.
        // old code: break after first evict_one, store still over cap.
        // fix: loop continues until own_foot <= cap, evicting all 3 .dts.
        const auto dcp = dirA / "cache.dcp";
        const auto dci = dirA / "cache.dci";
        write_file(dcp, 380u * 1024, now - std::chrono::hours(1));
        write_file(dci, 300u * 1024, now - std::chrono::hours(1));

        auto write_dts = [&](const fs::path & p, uint32_t tok_offset) {
            constexpr uint32_t N_TOK = 8;
            uint32_t hdr[3] = { DTS_MAGIC, DTS_VERSION, N_TOK };
            std::ofstream out(p, std::ios::binary);
            out.write((const char *) hdr, sizeof(hdr));
            uint64_t len[2] = { 0, 0 };
            uint32_t n_ckpt = 0;
            out.write((const char *) len, sizeof(len));
            out.write((const char *) &n_ckpt, sizeof(n_ckpt));
            uint64_t last_access = 0;
            out.write((const char *) &last_access, sizeof(last_access));
            for (uint32_t i = 0; i < N_TOK; i++) {
                int32_t t = (int32_t) (tok_offset + i);
                out.write((const char *) &t, sizeof(t));
            }
        };

        const auto dts1 = dirA / "a1.dts";
        const auto dts2 = dirA / "a2.dts";
        const auto dts3 = dirA / "a3.dts";
        write_dts(dts1, 100);
        write_dts(dts2, 200);
        write_dts(dts3, 300);

        fs::last_write_time(dts1, now - std::chrono::hours(3));
        fs::last_write_time(dts2, now - std::chrono::hours(2));
        fs::last_write_time(dts3, now - std::chrono::hours(1));

        write_meta(dirA, fingerprint);

        // own_foot = 0.6 MiB (store) + 192 B (.dts) ~= 0.6002 MiB > 0.6 MiB cap
        // -> knob B loop runs and should evict all 3 .dts
        std::vector<std::array<uint8_t, 32>> evicted;
        A.disk_prune_dpc(dirA.string(), &evicted);

        check(!file_exists(dts1), "s8: oldest .dts evicted (dts1)");
        check(!file_exists(dts2), "s8: middle .dts evicted (dts2)");
        check(!file_exists(dts3), "s8: newest .dts evicted (dts3)");
        check(evicted.size() == 3, "s8: all 3 fingerprints returned as evicted");
        // store files must remain (they are evicted only as last resort)
        check(file_exists(dcp) && file_exists(dci), "s8: store files kept after knob B eviction");
    }

    //
    //  scenario 9: chunk-store files are not pool-eviction candidates, even when
    //  the store alone exceeds the shared pool budget.
    //
    if (only.empty() || only == "s9") {
        fprintf(stderr, "\n== scenario 9: chunk-store files are retained ==\n");

        fs::remove_all(root / "preset-A");
        fs::remove_all(root / "preset-B");
        fs::create_directories(dirA);
        fs::create_directories(dirB);

        // The store alone exceeds the 1 MiB pool budget.
        const auto dcp = dirA / "cache.dcp";
        const auto dci = dirA / "cache.dci";
        write_file(dcp, 700u * 1024, now - std::chrono::hours(1));
        write_file(dci, 700u * 1024, now - std::chrono::hours(1));

        write_meta(dirA, fingerprint);

        // The policy prunes .dpc/.dts entries but never deletes the append-only
        // store files. The store can remain over budget until its own compaction.
        A.disk_prune_dpc(dirA.string(), nullptr);

        check(file_exists(dcp), "s9: .dcp is retained by pool pruning");
        check(file_exists(dci), "s9: .dci is retained by pool pruning");
    }

    if (only.empty() || only == "s10") {
        fprintf(stderr, "\n== scenario 10: cache save outcome distinguishes RAM from queued disk ==\n");
        const fs::path dir = root / "save-outcome";
        fs::create_directories(dir);
        const llama_tokens token_ids = { 1, 2, 3 };
        server_prompt prompt;
        prompt.tokens = server_tokens(token_ids, false);

        server_prompt_cache ram_cache(4, 0);
        auto * ram_state = ram_cache.alloc(prompt, 1024, 0);
        check(ram_state != nullptr, "s10: RAM state allocated");
        check(ram_cache.commit_disk_state(ram_state) == server_prompt_cache_save_result::stored_in_ram,
              "s10: RAM state reports stored");

        server_prompt_cache disk_cache(1, 0, 32);
        disk_cache.load_disk_index(dir.string(), fingerprint);
        auto * disk_state = disk_cache.alloc(prompt, 2u * 1024 * 1024, 0);
        check(disk_state != nullptr, "s10: oversized state allocated for write-through");
        check(disk_cache.commit_disk_state(disk_state) == server_prompt_cache_save_result::queued_to_disk,
              "s10: write-through reports queue admission, not durable storage");
        disk_cache.drain_disk_writes();
    }

    fs::remove_all(tmp, ec_tmp);

    if (g_failures) {
        fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }

    fprintf(stderr, "all ok\n");
    return 0;
}
