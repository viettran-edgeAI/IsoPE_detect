#pragma once

#include "../base/eml_base.h"
#include "../containers/STL_MCU.h"

#include <chrono>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

namespace eml {

    typedef enum TimeUnit : uint8_t {
        MICROSECONDS = 0,
        MILLISECONDS = 1,
        NANOSECONDS  = 2
    } TimeUnit;

    namespace detail_time {
        inline std::chrono::steady_clock::time_point program_start() {
            static const auto t0 = std::chrono::steady_clock::now();
            return t0;
        }
    }

    inline long unsigned eml_time_now(TimeUnit unit = TimeUnit::MILLISECONDS) {
        const auto elapsed = std::chrono::steady_clock::now() - detail_time::program_start();
        switch (unit) {
            case TimeUnit::MICROSECONDS:
                return static_cast<long unsigned>(
                    std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
            case TimeUnit::NANOSECONDS:
                return static_cast<long unsigned>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            case TimeUnit::MILLISECONDS:
            default:
                return static_cast<long unsigned>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        }
    }

    /**
     * @brief Return (resident_memory_bytes, 0) on Linux by parsing /proc/self/status.
     *        Falls back to (0, 0) on parse failure.
     */
    inline pair<size_t, size_t> eml_memory_status() {
        size_t rss_bytes = 0;
        std::ifstream status("/proc/self/status");
        if (status.is_open()) {
            std::string line;
            while (std::getline(status, line)) {
                if (line.rfind("VmRSS:", 0) == 0) {
                    // Format: "VmRSS:    <value> kB"
                    size_t kb = 0;
                    if (std::sscanf(line.c_str(), "VmRSS: %zu", &kb) == 1) {
                        rss_bytes = kb * 1024;
                    }
                    break;
                }
            }
        }
        return make_pair(rss_bytes, static_cast<size_t>(0));
    }

    typedef struct eml_time_anchor {
        long unsigned anchor_time;
        uint16_t index;
    } eml_time_anchor;

    template<typename BaseT, size_t PathBuffer = EML_PATH_BUFFER>
    class eml_logger_t {
        char time_log_path[PathBuffer] = {'\0'};
        char memory_log_path[PathBuffer] = {'\0'};
        vector<eml_time_anchor> time_anchors;

    public:
        uint32_t freeHeap = 0;
        uint32_t largestBlock = 0;
        long unsigned starting_time = 0;
        uint8_t fragmentation = 0;
        uint32_t lowest_ram = UINT32_MAX;
        uint64_t lowest_rom = UINT64_MAX;
        uint64_t freeDisk = 0;
        float log_time = 0.0f;

        void init(const BaseT* base, bool keep_old_file = false) {
            if (!base || !base->ready_to_use()) {
                eml_debug(1, "❌ Cannot init logger: base not ready");
                return;
            }

            time_anchors.clear();
            starting_time = eml_time_now(MILLISECONDS);
            drop_anchor();

            lowest_ram = UINT32_MAX;
            lowest_rom = UINT64_MAX;

            base->get_time_log_path(time_log_path);
            base->get_memory_log_path(memory_log_path);

            if (time_log_path[0] == '\0' || memory_log_path[0] == '\0') {
                eml_debug(1, "❌ Cannot init logger: log paths not set");
                return;
            }

            if (!keep_old_file) {
                if (std::filesystem::exists(time_log_path)) {
                    std::filesystem::remove(time_log_path);
                }
                {
                    std::ofstream logFile(time_log_path);
                    if (logFile.is_open()) {
                        logFile << "Event,\t\tTime(ms),duration,Unit\n";
                    }
                }

                if (std::filesystem::exists(memory_log_path)) {
                    std::filesystem::remove(memory_log_path);
                }
                {
                    std::ofstream memFile(memory_log_path);
                    if (memFile.is_open()) {
                        memFile << "Time(s),FreeHeap,Largest_Block,FreeDisk\n";
                    }
                }
            }

            t_log("init tracker");
            m_log("init tracker", true);
        }

        void m_log(const char* msg, bool log = true) {
            auto heap_status = eml_memory_status();
            freeHeap = static_cast<uint32_t>(heap_status.first);
            largestBlock = static_cast<uint32_t>(heap_status.second);

            // On Linux, query available filesystem space for the log directory
            freeDisk = 0;
            {
                std::error_code ec;
                auto si = std::filesystem::space(
                    std::filesystem::path(memory_log_path).parent_path(), ec);
                if (!ec) {
                    freeDisk = si.available;
                }
            }

            if (freeHeap < lowest_ram) lowest_ram = freeHeap;
            if (freeDisk < lowest_rom) lowest_rom = freeDisk;
            if (freeHeap > 0) {
                fragmentation = static_cast<uint8_t>(100 - (largestBlock * 100 / freeHeap));
            } else {
                fragmentation = 0;
            }

            if (log) {
                log_time = (eml_time_now(MILLISECONDS) - starting_time) / 1000.0f;
                std::ofstream logFile(memory_log_path, std::ios::app);
                if (logFile.is_open()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), "%.2f,\t%u,\t%u,\t%llu",
                                  log_time, freeHeap, largestBlock,
                                  (unsigned long long)freeDisk);
                    logFile << buf;
                    if (msg && std::strlen(msg) > 0) {
                        logFile << ",\t" << msg << "\n";
                    } else {
                        logFile << "\n";
                    }
                }
            }
        }

        void m_log() { m_log("", false); }

        uint16_t drop_anchor() {
            eml_time_anchor anchor;
            anchor.anchor_time = eml_time_now(MILLISECONDS);
            anchor.index = static_cast<uint16_t>(time_anchors.size());
            time_anchors.push_back(anchor);
            return anchor.index;
        }

        uint16_t current_anchor() const {
            return time_anchors.size() > 0 ? time_anchors.back().index : 0;
        }

        size_t memory_usage() const {
            return sizeof(*this);
        }

        long unsigned t_log(const char* msg, size_t begin_anchor_index, size_t end_anchor_index, const char* unit = "ms") {
            float ratio = 1.0f;
            if (std::strcmp(unit, "s") == 0 || std::strcmp(unit, "second") == 0) ratio = 1000.0f;
            else if (std::strcmp(unit, "us") == 0 || std::strcmp(unit, "microsecond") == 0) ratio = 0.001f;

            if (time_anchors.size() == 0) return 0;
            if (begin_anchor_index >= time_anchors.size() || end_anchor_index >= time_anchors.size()) return 0;
            if (end_anchor_index <= begin_anchor_index) {
                std::swap(begin_anchor_index, end_anchor_index);
            }

            const long unsigned begin_time = time_anchors[begin_anchor_index].anchor_time;
            const long unsigned end_time = time_anchors[end_anchor_index].anchor_time;
            const float elapsed = (end_time - begin_time) / ratio;

            {
                std::ofstream logFile(time_log_path, std::ios::app);
                if (logFile.is_open()) {
                    char buf[256];
                    if (msg && std::strlen(msg) > 0) {
                        std::snprintf(buf, sizeof(buf), "%s,\t%.1f,\t%.2f,\t%s\n",
                                      msg, begin_time / 1000.0f, elapsed, unit);
                    } else {
                        std::snprintf(buf, sizeof(buf), "unknown event,\t%.1f,\t%.2f,\t%s\n",
                                      begin_time / 1000.0f, elapsed, unit);
                    }
                    logFile << buf;
                }
            }

            time_anchors[end_anchor_index].anchor_time = eml_time_now(MILLISECONDS);
            return static_cast<long unsigned>(elapsed);
        }

        long unsigned t_log(const char* msg, size_t begin_anchor_index, const char* unit = "ms") {
            eml_time_anchor end_anchor;
            end_anchor.anchor_time = eml_time_now(MILLISECONDS);
            end_anchor.index = static_cast<uint16_t>(time_anchors.size());
            time_anchors.push_back(end_anchor);
            return t_log(msg, begin_anchor_index, end_anchor.index, unit);
        }

        long unsigned t_log(const char* msg) {
            const long unsigned current_time = eml_time_now(MILLISECONDS) - starting_time;
            {
                std::ofstream logFile(time_log_path, std::ios::app);
                if (logFile.is_open()) {
                    char buf[256];
                    if (msg && std::strlen(msg) > 0) {
                        std::snprintf(buf, sizeof(buf), "%s,\t%.1f,\t_,\tms\n",
                                      msg, current_time / 1000.0f);
                    } else {
                        std::snprintf(buf, sizeof(buf), "unknown event,\t%.1f,\t_,\tms\n",
                                      current_time / 1000.0f);
                    }
                    logFile << buf;
                }
            }
            return current_time;
        }
    };

} // namespace eml
