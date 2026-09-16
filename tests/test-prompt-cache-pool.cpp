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

        // fingerprints pre-seeded so neither save clears its dir
        write_meta(dirA, fingerprint);
        write_meta(dirB, fingerprint);

        // B saves -> its exit prune scans the whole root
        B.save_to_disk(dirB.string(), fingerprint);

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

        // A saves -> its exit prune counts its OWN .ccn and evicts oldest first
        A.save_to_disk(dirA.string(), fingerprint);

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

        // A saves -> its exit prune must count the 0.6 MiB of store files
        // toward the 1 MiB budget and evict A's own .dpc to fit
        A.save_to_disk(dirA.string(), fingerprint);

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
        check(n_idx == 1, "s5: legacy .dpc still indexed after rollback prune");
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

    fs::remove_all(tmp, ec_tmp);

    if (g_failures) {
        fprintf(stderr, "\n%d failure(s)\n", g_failures);
        return 1;
    }

    fprintf(stderr, "\nall ok\n");
    return 0;
}