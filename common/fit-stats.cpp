#include "fit-stats.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace fit_stats {

static void model_stat(const std::string & path, int64_t & mtime_sec, int64_t & size) {
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        mtime_sec = 0;
        size = 0;
        return;
    }
    auto ftime = fs::last_write_time(path, ec);
    if (!ec) {
        mtime_sec = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                ftime.time_since_epoch()).count();
    } else {
        mtime_sec = 0;
    }
    size = (int64_t) fs::file_size(path, ec);
    if (ec) {
        size = 0;
    }
}

std::string default_path() {
    const char * e = std::getenv("LLAMA_FIT_STATS");
    if (e && *e) {
        return std::string(e);
    }
    const char * home = std::getenv("HOME");
    std::string base = home ? std::string(home) : ".";
    return base + "/.cache/llama-cpp/fit-stats.txt";
}

static std::vector<std::string> split_tab(const std::string & line) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, '\t')) {
        f.push_back(tok);
    }
    return f;
}

std::map<std::string, int64_t> load_overheads(const std::string & model_path, const std::string & path) {
    std::map<std::string, int64_t> out;
    int64_t mt = 0, sz = 0;
    model_stat(model_path, mt, sz);

    std::ifstream in(path);
    if (!in) {
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> f = split_tab(line);
        if (f.size() != 5) {
            continue;
        }
        if (f[0] != model_path) {
            continue;
        }
        if (std::stoll(f[1]) != mt || std::stoll(f[2]) != sz) {
            continue;  // stale: model changed
        }
        out[f[3]] = std::stoll(f[4]);
    }
    return out;
}

void record_overhead(const std::string & model_path, const std::string & dev_name,
                     int64_t overhead_bytes, const std::string & path) {
    int64_t mt = 0, sz = 0;
    model_stat(model_path, mt, sz);

    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);

    // read existing lines, locate the (model, dev) entry to replace
    std::vector<std::string> lines;
    {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
    }

    std::string updated;
    {
        std::ostringstream o;
        o << model_path << '\t' << mt << '\t' << sz << '\t' << dev_name << '\t' << overhead_bytes;
        updated = o.str();
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return;
    }
    for (auto & line : lines) {
        std::vector<std::string> f = split_tab(line);
        if (f.size() == 5 && f[0] == model_path && f[3] == dev_name) {
            continue;  // drop old entry; replaced by `updated` below
        }
        out << line << '\n';
    }
    out << updated << '\n';
}

}  // namespace fit_stats
