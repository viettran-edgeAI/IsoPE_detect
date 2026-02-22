#pragma once
#include <iostream>
#include <iomanip>
#include <type_traits>

#ifndef RF_DEBUG_LEVEL
    #define RF_DEBUG_LEVEL 1
#else
    #if RF_DEBUG_LEVEL > 3
        #undef RF_DEBUG_LEVEL
        #define RF_DEBUG_LEVEL 3
    #endif
#endif

/*
 RF_DEBUG_LEVEL :
    0 : silent mode - no messages
    1 : forest messages (start, end, major events)
    2 : messages at components level + warnings
    3 : all memory and event timing messages & detailed info
 note: all error messages (lead to failed process) will be printed with RF_DEBUG_LEVEL >= 1
*/

inline void rf_debug_print_one(const char* msg) {
    std::cerr << msg;
}

template<typename T>
inline void rf_debug_print_one(const T& obj) {
    if constexpr (std::is_floating_point_v<T>) {
        std::cerr << std::fixed << std::setprecision(3) << obj;
        std::cerr.unsetf(std::ios::fixed);
    } else {
        std::cerr << obj;
    }
}

inline void rf_debug_print() {
    std::cerr << '\n';
}

template<typename... Args>
inline void rf_debug_print(const Args&... args) {
#if RF_DEBUG_LEVEL > 0
    (rf_debug_print_one(args), ...);
    std::cerr << '\n';
#endif
}

#define eml_debug(level, ...)                         \
    do{                                               \
        if constexpr (RF_DEBUG_LEVEL > (level)) {     \
            rf_debug_print(__VA_ARGS__);              \
        }                                             \
    }while(0)

#define eml_debug_2(level, msg1, obj1, msg2, obj2)    \
    eml_debug(level, msg1, obj1, " ", msg2, obj2)
