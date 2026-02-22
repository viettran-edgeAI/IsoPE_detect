#pragma once

#include "../containers/STL_MCU.h"
#include "../../Rf_file_manager.h"
#include "../../Rf_board_config.h"

#if defined(ESP_PLATFORM)
    #include "esp_system.h"
    #if RF_BOARD_SUPPORTS_PSRAM
    #include <esp_psram.h>
    #endif
#endif

#include <cstddef>
#include <cstring>

namespace mcu {

    typedef enum TimeUnit : uint8_t {
        MICROSECONDS = 0,
        MILLISECONDS = 1,
        NANOSECONDS  = 2
    } TimeUnit;

    inline long unsigned eml_time_now(TimeUnit unit = TimeUnit::MILLISECONDS) {
        return (unit == TimeUnit::MICROSECONDS) ? static_cast<long unsigned>(micros())
                                                     : static_cast<long unsigned>(millis());
    }

    inline pair<size_t, size_t> eml_memory_status() {
        size_t freeHeap = 0;
        size_t largestBlock = 0;

        #if defined(ESP_PLATFORM)
            const uint32_t internalCaps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
            #if RF_PSRAM_AVAILABLE && RF_USE_PSRAM
                if (esp_psram_is_initialized()) {
                    freeHeap = static_cast<size_t>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
                    largestBlock = static_cast<size_t>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
                } else {
                    freeHeap = static_cast<size_t>(heap_caps_get_free_size(internalCaps));
                    largestBlock = static_cast<size_t>(heap_caps_get_largest_free_block(internalCaps));
                }
            #else
                freeHeap = static_cast<size_t>(heap_caps_get_free_size(internalCaps));
                largestBlock = static_cast<size_t>(heap_caps_get_largest_free_block(internalCaps));
            #endif
        #endif

        return make_pair(freeHeap, largestBlock);
    }

    typedef struct eml_time_anchor {
        long unsigned anchor_time;
        uint16_t index;
    } eml_time_anchor;

    template<typename BaseT, size_t PathBuffer = EML_PATH_BUFFER>
    class eml_logger_t {
        char time_log_path[PathBuffer] = {'\0'};
        char memory_log_path[PathBuffer] = {'\0'};
        b_vector<eml_time_anchor> time_anchors;

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
                if (RF_FS_EXISTS(time_log_path)) {
                    RF_FS_REMOVE(time_log_path);
                }
                File logFile = RF_FS_OPEN(time_log_path, FILE_WRITE);
                if (logFile) {
                    logFile.println("Event,\t\tTime(ms),duration,Unit");
                    logFile.close();
                }

                if (RF_FS_EXISTS(memory_log_path)) {
                    RF_FS_REMOVE(memory_log_path);
                }
                File memFile = RF_FS_OPEN(memory_log_path, FILE_WRITE);
                if (memFile) {
                    memFile.println("Time(s),FreeHeap,Largest_Block,FreeDisk");
                    memFile.close();
                }
            }

            t_log("init tracker");
            m_log("init tracker", true);
        }

        void m_log(const char* msg, bool log = true) {
            auto heap_status = eml_memory_status();
            freeHeap = heap_status.first;
            largestBlock = heap_status.second;

            const uint64_t totalBytes = RF_TOTAL_BYTES();
            const uint64_t usedBytes = RF_USED_BYTES();
            freeDisk = (totalBytes >= usedBytes) ? (totalBytes - usedBytes) : 0;

            if (freeHeap < lowest_ram) lowest_ram = freeHeap;
            if (freeDisk < lowest_rom) lowest_rom = freeDisk;
            if (freeHeap > 0) {
                fragmentation = 100 - (largestBlock * 100 / freeHeap);
            } else {
                fragmentation = 0;
            }

            if (log) {
                log_time = (eml_time_now(MILLISECONDS) - starting_time) / 1000.0f;
                File logFile = RF_FS_OPEN(memory_log_path, FILE_APPEND);
                if (logFile) {
                    logFile.printf("%.2f,\t%u,\t%u,\t%llu",
                                 log_time, freeHeap, largestBlock, (unsigned long long)freeDisk);
                    if (msg && strlen(msg) > 0) {
                        logFile.printf(",\t%s\n", msg);
                    } else {
                        logFile.println();
                    }
                    logFile.close();
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
            if (strcmp(unit, "s") == 0 || strcmp(unit, "second") == 0) ratio = 1000.0f;
            else if (strcmp(unit, "us") == 0 || strcmp(unit, "microsecond") == 0) ratio = 0.001f;

            if (time_anchors.size() == 0) return 0;
            if (begin_anchor_index >= time_anchors.size() || end_anchor_index >= time_anchors.size()) return 0;
            if (end_anchor_index <= begin_anchor_index) {
                std::swap(begin_anchor_index, end_anchor_index);
            }

            const long unsigned begin_time = time_anchors[begin_anchor_index].anchor_time;
            const long unsigned end_time = time_anchors[end_anchor_index].anchor_time;
            const float elapsed = (end_time - begin_time) / ratio;

            File logFile = RF_FS_OPEN(time_log_path, FILE_APPEND);
            if (logFile) {
                if (msg && strlen(msg) > 0) {
                    logFile.printf("%s,\t%.1f,\t%.2f,\t%s\n", msg, begin_time / 1000.0f, elapsed, unit);
                } else {
                    logFile.printf("unknown event,\t%.1f,\t%.2f,\t%s\n", begin_time / 1000.0f, elapsed, unit);
                }
                logFile.close();
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
            File logFile = RF_FS_OPEN(time_log_path, FILE_APPEND);
            if (logFile) {
                if (msg && strlen(msg) > 0) {
                    logFile.printf("%s,\t%.1f,\t_,\tms\n", msg, current_time / 1000.0f);
                } else {
                    logFile.printf("unknown event,\t%.1f,\t_,\tms\n", current_time / 1000.0f);
                }
                logFile.close();
            }
            return current_time;
        }
    };

} // namespace mcu
