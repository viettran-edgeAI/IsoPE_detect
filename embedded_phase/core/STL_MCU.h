#pragma once

#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <cassert>
#include <utility>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "hash_kernel.h"
#include "initializer_list.h"

#define hashers best_hashers_16 // change to best_hashers_8 to save 255 bytes of disk space, but more collisions

namespace mcu {
    namespace detail_simd {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        [[gnu::target("avx2")]] inline void fill_words_avx2(size_t* dst, size_t word_count, size_t value) noexcept {
            const __m256i v = _mm256_set1_epi64x(static_cast<long long>(value));
            size_t i = 0;
            for (; i + 4 <= word_count; i += 4) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
            }
            for (; i < word_count; ++i) {
                dst[i] = value;
            }
        }

        [[gnu::target("avx2")]] inline size_t first_nonzero_word_avx2(const size_t* data, size_t word_count) noexcept {
            size_t i = 0;
            for (; i + 4 <= word_count; i += 4) {
                const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                const __m256i zero = _mm256_setzero_si256();
                const __m256i eq = _mm256_cmpeq_epi64(block, zero);
                const int zero_mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq));
                if (zero_mask != 0xF) {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        if (data[i + lane] != 0) {
                            return i + lane;
                        }
                    }
                }
            }
            for (; i < word_count; ++i) {
                if (data[i] != 0) {
                    return i;
                }
            }
            return std::numeric_limits<size_t>::max();
        }

        [[gnu::target("avx2")]] inline size_t last_nonzero_word_avx2(const size_t* data, size_t word_count) noexcept {
            size_t i = word_count;
            while (i >= 4) {
                const size_t block_start = i - 4;
                const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + block_start));
                const __m256i zero = _mm256_setzero_si256();
                const __m256i eq = _mm256_cmpeq_epi64(block, zero);
                const int zero_mask = _mm256_movemask_pd(_mm256_castsi256_pd(eq));
                if (zero_mask != 0xF) {
                    for (size_t lane = 4; lane > 0; --lane) {
                        const size_t idx = block_start + lane - 1;
                        if (data[idx] != 0) {
                            return idx;
                        }
                    }
                }
                i -= 4;
            }

            for (size_t j = i; j > 0; --j) {
                if (data[j - 1] != 0) {
                    return j - 1;
                }
            }
            return std::numeric_limits<size_t>::max();
        }
#endif

        inline void fill_words(size_t* dst, size_t word_count, size_t value) noexcept {
            if (!dst || word_count == 0) {
                return;
            }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (__builtin_cpu_supports("avx2") && word_count >= 16) {
                fill_words_avx2(dst, word_count, value);
                return;
            }
#endif
            std::fill_n(dst, word_count, value);
        }

        inline size_t first_nonzero_word(const size_t* data, size_t word_count) noexcept {
            if (!data || word_count == 0) {
                return std::numeric_limits<size_t>::max();
            }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (__builtin_cpu_supports("avx2") && word_count >= 16) {
                return first_nonzero_word_avx2(data, word_count);
            }
#endif
            for (size_t i = 0; i < word_count; ++i) {
                if (data[i] != 0) {
                    return i;
                }
            }
            return std::numeric_limits<size_t>::max();
        }

        inline size_t last_nonzero_word(const size_t* data, size_t word_count) noexcept {
            if (!data || word_count == 0) {
                return std::numeric_limits<size_t>::max();
            }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (__builtin_cpu_supports("avx2") && word_count >= 16) {
                return last_nonzero_word_avx2(data, word_count);
            }
#endif
            for (size_t i = word_count; i > 0; --i) {
                if (data[i - 1] != 0) {
                    return i - 1;
                }
            }
            return std::numeric_limits<size_t>::max();
        }

        inline unsigned trailing_zeros(size_t value) noexcept {
#if SIZE_MAX > 0xFFFFFFFFu
            return static_cast<unsigned>(__builtin_ctzll(static_cast<unsigned long long>(value)));
#else
            return static_cast<unsigned>(__builtin_ctzl(static_cast<unsigned long>(value)));
#endif
        }

        inline unsigned leading_zeros(size_t value) noexcept {
#if SIZE_MAX > 0xFFFFFFFFu
            return static_cast<unsigned>(__builtin_clzll(static_cast<unsigned long long>(value)));
#else
            return static_cast<unsigned>(__builtin_clzl(static_cast<unsigned long>(value)));
#endif
        }
    }
    
    template<typename T1, typename T2>
    struct pair {
        T1 first;
        T2 second;

        // Trivial default, copy and move
        [[gnu::always_inline]] constexpr pair() noexcept = default;
        [[gnu::always_inline]] constexpr pair(const T1& a, const T2& b) noexcept : first(a), second(b) {}
        [[gnu::always_inline]] constexpr pair(T1&& a, T2&& b) noexcept : first(std::move(a)), second(std::move(b)) {}

        // Allow initialization from different pair types
        template<typename U1, typename U2>
        [[gnu::always_inline]] constexpr pair(const pair<U1, U2>& p) noexcept : first(p.first), second(p.second) {}

        template<typename U1, typename U2>
        [[gnu::always_inline]] constexpr pair(pair<U1, U2>&& p) noexcept : first(std::move(p.first)), second(std::move(p.second)) {}

        [[gnu::always_inline]] constexpr pair(const pair&) noexcept = default;
        [[gnu::always_inline]] constexpr pair(pair&&) noexcept = default;

        [[gnu::always_inline]] constexpr pair& operator=(const pair&) noexcept = default;
        [[gnu::always_inline]] constexpr pair& operator=(pair&&) noexcept = default;

        // Allow assignment from different pair types
        template<typename U1, typename U2>
        [[gnu::always_inline]] constexpr pair& operator=(const pair<U1, U2>& p) noexcept {
            first = p.first;
            second = p.second;
            return *this;
        }

        template<typename U1, typename U2>
        [[gnu::always_inline]] constexpr pair& operator=(pair<U1, U2>&& p) noexcept {
            first = std::move(p.first);
            second = std::move(p.second);
            return *this;
        }

        // Comparisons - marked [[nodiscard]] and [[gnu::pure]]
        [[nodiscard, gnu::pure]] constexpr bool operator==(const pair& o) const noexcept {
            return first == o.first && second == o.second;
        }
        [[nodiscard, gnu::pure]] constexpr bool operator!=(const pair& o) const noexcept {
            return !(*this == o);
        }

        // Optimized ordering comparisons
        [[nodiscard, gnu::pure]] constexpr bool operator<(const pair& o) const noexcept {
            return first < o.first || (!(o.first < first) && second < o.second);
        }
        [[nodiscard, gnu::pure]] constexpr bool operator<=(const pair& o) const noexcept {
            return !(o < *this);
        }
        [[nodiscard, gnu::pure]] constexpr bool operator>(const pair& o) const noexcept {
            return o < *this;
        }
        [[nodiscard, gnu::pure]] constexpr bool operator>=(const pair& o) const noexcept {
            return !(*this < o);
        }

        // In-class make_pair for better locality and potential inlining
        [[gnu::always_inline]] static inline constexpr pair<T1, T2> make_pair(const T1& a, const T2& b) noexcept {
            return pair<T1, T2>(a, b);
        }
        [[gnu::always_inline]] static inline constexpr pair<T1, T2> make_pair(T1&& a, T2&& b) noexcept {
            return pair<T1, T2>(std::move(a), std::move(b));
        }
    };

    // Global make_pair for API compatibility
    template<typename T1, typename T2>
    [[gnu::always_inline]] inline constexpr pair<std::decay_t<T1>, std::decay_t<T2>> make_pair(T1&& a, T2&& b) noexcept {
        return pair<std::decay_t<T1>, std::decay_t<T2>>(std::forward<T1>(a), std::forward<T2>(b));
    }

    // unordered_map_s class : for speed and flexibility, but limited to small number of elements (max 255)
    template<typename V, typename T>
    class unordered_map_s : public hash_kernel, public slot_handler {
    private:
        using Pair = pair<V, T>;

        Pair* table = nullptr;
        uint8_t size_ = 0;
        uint8_t dead_size_ = 0;     // used + tombstones 
        uint8_t fullness_ = 92; //(%)       . virtual_cap = cap_ * fullness_ / 100
        uint8_t virtual_cap = 0; // virtual capacity
        uint8_t step_ = 0;
        // cap_        : for internal use 
        // virtual_cap : for user
        // ----------------------------------------------  : table size
        // --------------|--------------------|----------|
        //             size_             virtual_cap    cap_
        static T MAP_DEFAULT_VALUE;

        bool rehash(uint8_t newCap) noexcept {
            if (newCap < size_) newCap = size_;
            if (newCap > MAX_CAP) newCap = MAX_CAP;
            if (newCap == 0) newCap = INIT_CAP;

            Pair* newTable = new (std::nothrow) Pair[newCap];
            if (!newTable) {
                return false;
            }

            const size_t flagBytes = (static_cast<size_t>(newCap) * 2 + 7) / 8;
            uint8_t* newFlags = new (std::nothrow) uint8_t[flagBytes];
            if (!newFlags) {
                delete[] newTable;
                return false;
            }
            std::fill_n(newFlags, flagBytes, static_cast<uint8_t>(0));

            Pair* oldTable = table;
            uint8_t* oldFlags = flags;
            uint8_t oldCap = cap_;

            table = newTable;
            flags = newFlags;
            cap_ = newCap;
            size_ = 0;
            dead_size_ = 0;
            virtual_cap = cap_to_virtual();
            step_ = calStep(newCap);

            if (oldTable && oldFlags) {
                for (uint8_t i = 0; i < oldCap; ++i) {
                    if (getStateFrom(oldFlags, i) == slotState::Used) {
                        insert(oldTable[i]);
                    }
                }
            }

            if (oldFlags) {
                delete[] oldFlags;
            }
            delete[] oldTable;
            return true;
        }
        // safely convert between cap_ and virtual_cap 
        // ensure integrity after 2-way conversion
        uint8_t cap_to_virtual() const noexcept {
            return static_cast<uint8_t>((static_cast<uint16_t>(cap_) * fullness_) / 100);
        }
        
        uint8_t virtual_to_cap(uint8_t v_cap) const noexcept {
            return static_cast<uint8_t>((static_cast<uint16_t>(v_cap) * 100) / fullness_);
        }

        bool inline is_full() const noexcept {
            return size_ >= virtual_cap;
        }
        template<typename U, typename R> friend class unordered_map;
        template<typename U> friend class unordered_set;
    protected:
        int16_t getValue(V key) noexcept {
            if (cap_ == 0 || table == nullptr) {
                return -1;
            }
            uint8_t index     = hashFunction(cap_, key, hashers[cap_ - 1]);
            uint8_t attempts= 0;

            while(getState(index) != slotState::Empty) {
                slotState st = getState(index);
                if (attempts++ == cap_) {
                    break;
                }
                if (table[index].first == key) {
                    if (st == slotState::Used) {
                        // existing element
                        return table[index].second;
                    }
                    if (st == slotState::Deleted) {
                        // reuse this tombstone
                        break;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return -1;
        }
        int16_t getValue(V key) const noexcept {
            if (cap_ == 0 || table == nullptr) {
                return -1;
            }
            uint8_t index     = hashFunction(cap_, key, hashers[cap_ - 1]);
            uint8_t attempts= 0;

            while(getState(index) != slotState::Empty) {
                slotState st = getState(index);
                if (attempts++ == cap_) {
                    break;
                }
                if (table[index].first == key) {
                    if (st == slotState::Used) {
                        // existing element
                        return table[index].second;
                    }
                    if (st == slotState::Deleted) {
                        // reuse this tombstone
                        break;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return -1;
        }

        
    public:
        // default constructor
        unordered_map_s() noexcept {
            rehash(4);
        }

        /**
         * @brief Constructor with specified initial capacity.
         * @param cap Initial capacity (number of elements) the map should accommodate.
         */
        explicit unordered_map_s(uint8_t cap) noexcept {
            rehash(cap);
        }
        // destructor
        ~unordered_map_s() noexcept {
            delete[] table;
        }

        /**
         * @brief Copy constructor, creates a deep copy of another map.
         * @param other The map to copy from.
         */
        unordered_map_s(const unordered_map_s& other) noexcept : hash_kernel(),
            slot_handler(other),        
            size_(other.size_),
            dead_size_(other.dead_size_),
            fullness_(other.fullness_),
            virtual_cap(other.virtual_cap),
            step_(other.step_)
        {
            table = new (std::nothrow) Pair[cap_];
            for (uint8_t i = 0; i < cap_; ++i) {
                if (getState(i) == slotState::Used)
                    table[i] = other.table[i];
            }
        }


        /**
         * @brief Move constructor, transfers ownership of resources.
         * @param other The map to move from (will be left in a valid but unspecified state).
         */
        unordered_map_s(unordered_map_s&& other) noexcept : hash_kernel(),
        slot_handler(std::move(other)),  // ← steal flags & cap_
        size_(other.size_),
        dead_size_(other.dead_size_),
        fullness_(other.fullness_),
        virtual_cap(other.virtual_cap),
        step_(other.step_)
        {
            table       = other.table;
            other.table = nullptr;
            other.size_ = 0;
            other.dead_size_ = 0;
            other.fullness_ = 92;
            other.virtual_cap = 0;
            other.step_ = 0;
        }

        /**
         * @brief Copy assignment operator, replaces contents with a copy of another map.
         * @param other The map to copy from.
         * @return Reference to *this.
         */
        unordered_map_s& operator=(const unordered_map_s& other) noexcept {
            if (this != &other) {
                delete[] table;
                slot_handler::operator=(other);  // copy flags & cap_
                size_      = other.size_;
                fullness_  = other.fullness_;
                virtual_cap = other.virtual_cap;
                step_      = other.step_;
                table      = new (std::nothrow) Pair[cap_];
                for (uint8_t i = 0; i < cap_; ++i) {
                    if (getState(i) == slotState::Used) {
                        table[i] = other.table[i];
                    }
                }
            }
            return *this;
        }

        /**
         * @brief Move assignment operator, transfers ownership of resources.
         * @param other The map to move from (will be left in a valid but unspecified state).
         * @return Reference to *this.
         */
        unordered_map_s& operator=(unordered_map_s&& other) noexcept {
            if (this != &other) {
                delete[] table;
                slot_handler::operator=(std::move(other)); // steal flags & cap_
                size_      = other.size_;
                dead_size_ = other.dead_size_;
                fullness_  = other.fullness_;
                table      = other.table;
                virtual_cap = other.virtual_cap;
                step_      = other.step_;
                // reset other
                other.table = nullptr;
                other.size_ = 0;
                other.dead_size_ = 0;
                other.cap_  = 0;
                other.fullness_ = 92;
                other.virtual_cap = 0;
                other.step_ = 0;
            }
            return *this;
        }

        // Iterator traits for STL compatibility
        template<bool IsConst>
        class base_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = typename std::conditional<IsConst, const Pair, Pair>::type;
            using pointer           = typename std::conditional<IsConst, const Pair*, Pair*>::type;
            using reference         = value_type&;

        private:
            typename std::conditional<IsConst, const unordered_map_s*, unordered_map_s*>::type map_;
            uint8_t index_;

            void advance() {
                while (index_ < map_->cap_ && map_->getState(index_) != slotState::Used)
                    ++index_;
            }
            void retreat() {
                if (index_ == 0) return;
                uint8_t i = index_ - 1;
                while (i > 0 && map_->getState(i) != slotState::Used) --i;
                if (map_->getState(i) == slotState::Used) index_ = i;
            }

        public:
            // zero-arg ctor for “end()” 
            constexpr base_iterator() noexcept
                : map_(nullptr), index_(MAX_CAP)
            {}
        
            // your existing two-arg ctor
            base_iterator(decltype(map_) m, uint8_t start) noexcept
                : map_(m), index_(start)
            {
                if (map_) advance();
            }
            base_iterator& operator++()    { ++index_; advance(); return *this; }
            base_iterator  operator++(int) { auto tmp=*this; ++*this; return tmp; }
            base_iterator& operator--()    { retreat(); return *this; }
            base_iterator  operator--(int) { auto tmp=*this; --*this; return tmp; }

            reference operator*()  const { return map_->table[index_]; }
            pointer   operator->() const { return &map_->table[index_]; }

            bool operator==(const base_iterator& o) const {
                return map_==o.map_ && index_==o.index_;
            }
            bool operator!=(const base_iterator& o) const {
                return !(*this==o);
            }
        };

        using iterator       = base_iterator<false>;
        using const_iterator = base_iterator<true>;

        /**
         * @brief Returns an iterator to the beginning of the map.
         * @return Iterator pointing to the first element.
         */
        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, cap_);}

        const_iterator cbegin() const { return const_iterator(this, 0); }
        const_iterator cend() const { return const_iterator(this, cap_); }
        const_iterator begin() const { return cbegin(); }
        const_iterator end() const { return cend(); }

    private:
        pair<iterator, bool> insert_core(Pair&& p) noexcept {
            if (dead_size_ >= virtual_cap) {
                if (size_ == map_ability())
                    return { end(), false };
                uint16_t dbl = cap_ ? cap_ * 2: INIT_CAP;
                if (dbl > MAX_CAP) dbl = MAX_CAP;
                if (!rehash(static_cast<uint8_t>(dbl))) {
                    return { end(), false };
                }
            }

            if (cap_ == 0 || table == nullptr) {
                return { end(), false };
            }

            V key       = p.first;
            uint8_t index     = hashFunction(cap_, key, hashers[cap_ - 1]);

            while (getState(index) != slotState::Empty) {
                slotState st = getState(index);
                if (table[index].first == key) {
                    if (st == slotState::Used) {
                        // existing element
                        return { iterator(this, index), false };
                    }
                    if (st == slotState::Deleted) {         // reuse this tombstone
                        break;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }

            slotState old_state = getState(index);
            table[index] = std::move(p);
            setState(index, slotState::Used);
            ++size_;
            if (old_state == slotState::Empty) {
                ++dead_size_;  
            }
            return { iterator(this, index), true };
        }
    public:
        /**
         * @brief Inserts a pair into the map, copying if needed.
         * @param p The key-value pair to insert.
         * @return A pair containing an iterator to the inserted element and a boolean indicating whether insertion took place.
         */
        pair<iterator,bool> insert(const Pair& p) noexcept {
            return insert_core(Pair(p));  // Copy ilue and move
        }

        /**
         * @brief Inserts a pair into the map using move semantics.
         * @param p The key-value pair to insert.
         * @return A pair containing an iterator to the inserted element and a boolean indicating whether insertion took place.
         */
        pair<iterator, bool> insert(Pair&& p) noexcept {
            return insert_core(std::move(p));  // Move directly
        }

        /**
         * @brief Inserts a key-value pair into the map, with perfect forwarding.
         * @param key The key to insert.
         * @param value The value to insert, forwarded to avoid extra copies.
         * @return A pair containing an iterator to the inserted element and a boolean indicating whether insertion took place.
         */
        template<typename U>
        pair<iterator,bool> insert(V key, U&& value) noexcept {
            return insert_core(Pair(key, std::forward<U>(value)));
        }

        /**
         * @brief Removes an element with the specified key.
         * @param key The key to find and remove.
         * @return true if an element was removed, false otherwise.
         */
        bool erase(V key) noexcept {
            if (cap_ == 0 || table == nullptr) {
                return false;
            }
            uint8_t index = hashFunction(cap_, key, hashers[cap_ - 1]);
            uint8_t attempt = 0;

            while (getState(index) != slotState::Empty) {
                if(attempt++ == cap_) return false;
                if (table[index].first == key) {
                    if (getState(index) == slotState::Used) {
                        setState(index, slotState::Deleted);
                        --size_;
                        // note : consider rehash when there are too many tombstones in map
                        return true;
                    } else if (getState(index) == slotState::Deleted) {
                        return false;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return false;
        }

        /**
         * @brief Finds an element with the specified key.
         * @param key The key to find.
         * @return Iterator to the element if found, otherwise end().
         */
        iterator find(V key) noexcept {
            if (cap_ == 0 || table == nullptr) {
                return end();
            }
            uint8_t index = hashFunction(cap_, key, hashers[cap_ - 1]);
            uint8_t attempt = 0;
            // Search for a cell whose is used and matched key
            slotState st = getState(index);
            while (st != slotState::Empty) {
                if (attempt++ == cap_){
                    return end();
                }
                st = getState(index);
                if(table[index].first == key){
                    if(st == slotState::Used){
                        return iterator(this, index);
                    }
                    else if(st == slotState::Deleted){
                        return end();
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return end();
        }


        /**
         * @brief Access or insert an element.
         * @param key The key to find or insert.
         * @return Reference to the mapped value at the specified key.
         * @note If the key does not exist, a new element with default-constructed value is inserted.
         */
        T& operator[](V key) noexcept {
            iterator it = find(key);
            if (it != end()) {
                return it->second;
            } else {
                return insert(key, T()).first->second; // Insert and return reference
            }
        }

        /**
         * @brief Access an element with bounds checking.
         * @param key The key to find.
         * @return Reference to the mapped value at the specified key.
         * @note Returns MAP_DEFAULT_VALUE if key is not found.
         */
        T& at(V key) noexcept {
            iterator it = find(key);
            if (it != end()) {
                return it->second;
            } else {
                return MAP_DEFAULT_VALUE; // or throw an exception
                // throw std::out_of_range("key not found !");
            }
        }
        /**
         * @brief Checks if this map contains the same elements as another.
         * @param other The map to compare with.
         * @return true if maps are equal, false otherwise.
         */
        bool operator==(const unordered_map_s& other) const noexcept {
            if (size_ != other.size_) return false;
            for (uint8_t i = 0; i < cap_; ++i) {
                if (getState(i) == slotState::Used && !other.contains(table[i].first)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Checks if this map differs from another.
         * @param other The map to compare with.
         * @return true if maps are not equal, false otherwise.
         */
        bool operator!=(const unordered_map_s& other) const noexcept {
            return !(*this == other);
        }

        /**
         * @brief Gets the current fullness factor.
         * @return The current fullness factor as a float (0.0 to 1.0).
         */
        float get_fullness() const noexcept {
            return static_cast<float>(fullness_) / 100.0f;
        }

        /**
         * @brief Sets the fullness factor for the map.
         * @param fullness The new fullness factor (0.1 to 1.0 or 10 to 100).
         * @return true if successful, false if the new fullness would overflow the map.
         * @note Lower fullness reduces collisions but increases memory usage:
         *       0.9 -> -71% collisions | +11% memory
         *       0.8 -> -87% collisions | +25% memory
         *       0.7 -> -94% collisions | +43% memory
         */
        bool set_fullness(float fullness) noexcept {
            // Ensure fullness is within the valid range [0.1, 1.0] or [10, 100] (% format)
            if(fullness < 0.1f) fullness = 0.1f;
            if(fullness > 1.0f && fullness < 10) fullness = 1.0f;
            if(fullness > 100) fullness = 100;

            uint8_t old_fullness_ = fullness_;

            if(fullness <= 1.0f){
                fullness_ = static_cast<uint8_t>(fullness * 100);
            }else{
                fullness_ = static_cast<uint8_t>(fullness);
            }
            if(map_ability() < size_){
                fullness_ = old_fullness_;
                return false;
            }
            virtual_cap = cap_to_virtual();
            return true;
        }

        /**
         * @brief Checks if the map contains an element with the specified key.
         * @param key The key to find.
         * @return true if the key exists, false otherwise.
         */
        bool contains(V key) noexcept {
            return find(key) != end();
        }
        
        /**
         * @brief Shrinks the map's capacity to fit its size.
         * @return Number of bytes freed by shrinking.
         */
        size_t shrink_to_fit() noexcept {
            if (size_ < cap_) {
                const uint8_t oldCap = cap_;
                const size_t oldFlagBytes = (static_cast<size_t>(oldCap) * 2 + 7) / 8;

                uint8_t target_buckets = std::max<uint8_t>(
                    (size_ * 100 + fullness_ - 1) / fullness_, INIT_CAP);
                if (!rehash(target_buckets)) {
                    return 0;
                }
                // Calculate bytes saved:
                size_t tableSaved = (oldCap > cap_) ?
                    static_cast<size_t>(oldCap - cap_) * sizeof(Pair) : 0;
                size_t flagsSaved = (oldFlagBytes > ((static_cast<size_t>(cap_) * 2 + 7) / 8)) ?
                    oldFlagBytes - ((static_cast<size_t>(cap_) * 2 + 7) / 8) : 0;
                return tableSaved + flagsSaved;
            }
            return 0;
        }
        
        /**
         * @brief Removes all elements from the map.
         * @note Keeps the allocated memory for reuse.
         */
        void clear() noexcept {
            if (!flags) {
                size_ = 0;
                dead_size_ = 0;
                return;
            }
            std::fill_n(flags, (static_cast<size_t>(cap_) * 2 + 7) / 8, static_cast<uint8_t>(0));
            size_ = 0;
            dead_size_ = 0;
        }

        /**
         * @brief Reserves space for a specified number of elements.
         * @param new_virtual_cap The new virtual capacity to reserve.
         * @return true if successful, false if the requested capacity is too large.
         * @note This prepares the map to hold the specified number of elements without rehashing.
         */
        bool reserve(uint8_t new_virtual_cap) noexcept {
            // uint8_t newCap = static_cast<uint8_t>(cap_ * (100.0f / fullness_));
            uint8_t newCap = virtual_to_cap(new_virtual_cap);
            if (newCap > MAX_CAP) return false;
            if (newCap < size_) newCap = size_;
            if (newCap == cap_) return true;
            return rehash(newCap);
        }

        /**
         * @brief Gets the maximum theoretical number of elements the map can hold.
         * @return Maximum capacity based on the current fullness setting.
         */
        uint16_t map_ability() const noexcept {
            return static_cast<uint16_t>(MAX_CAP) * fullness_ / 100;
        }
        /**
         * @brief Gets the current number of elements in the map.
         * @return The element count.
         */
        uint16_t size() const noexcept { return size_; }

        /**
         * @brief Gets the current virtual capacity of the map.
         * @return The current virtual capacity.
         */
        uint16_t capacity() const noexcept { 
            return virtual_cap;
        }

        /**
         * @brief Checks if the map is empty.
         * @return true if the map contains no elements, false otherwise.
         */
        bool empty() const noexcept { return size_ == 0; }

        /**
         * @brief Calculates the total memory usage of the map.
         * @return Memory usage in bytes.
         * @note Includes the map object, elements table, and flag array.
         */
        size_t memory_usage() const noexcept {
            size_t table_bytes = static_cast<size_t>(cap_) * sizeof(Pair);
            size_t flags_bytes = (cap_ * 2 + 7) / 8;
            return sizeof(*this) + table_bytes + flags_bytes;
        }

        /**
         * @brief Checks if the table pointer is allocated in PSRAM.
         * @return true if table is in PSRAM, false if in DRAM or null.
         */
        bool is_table_in_psram() const noexcept {
            return false;
        }

        /**
         * @brief Gets the table pointer for debugging (returns nullptr if not allocated).
         * @return Pointer to the internal table (may be in PSRAM or DRAM).
         */
        const Pair* get_table_ptr() const noexcept {
            return table;
        }

        /**
         * @brief Swaps the contents of two maps.
         * @param other The map to swap with.
         */
        void swap(unordered_map_s& other) noexcept {
            std::swap(table, other.table);
            std::swap(flags, other.flags);
            std::swap(cap_, other.cap_);
            std::swap(size_, other.size_);
            std::swap(dead_size_, other.dead_size_);
            std::swap(fullness_, other.fullness_);
            std::swap(virtual_cap, other.virtual_cap);
            std::swap(step_, other.step_);
        }
    };
    template<typename V, typename T>
    T unordered_map_s<V, T>::MAP_DEFAULT_VALUE = T(); // Default value for MAP_DEFAULT_VALUE
    /*
    ------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------- unordered_set_s -----------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    template<typename T>
    class unordered_set_s : public hash_kernel, public slot_handler {
    private:
        T* table = nullptr;
        uint8_t size_ = 0;
        uint8_t dead_size_ = 0;     // used + tombstones
        uint8_t fullness_ = 92; //(%)       . virtual_cap = cap_ * fullness_ / 100
        uint8_t virtual_cap = 0; // virtual capacity
        uint8_t step_ = 0;

        bool rehash(uint8_t newCap) noexcept {
            if (newCap < size_) newCap = size_;
            if (newCap > MAX_CAP) newCap = MAX_CAP;
            if (newCap == 0) newCap = INIT_CAP;

            T* newTable = new (std::nothrow) T[newCap];
            if (!newTable) {
                return false;
            }

            const size_t flagBytes = (static_cast<size_t>(newCap) * 2 + 7) / 8;
            uint8_t* newFlags = new (std::nothrow) uint8_t[flagBytes];
            if (!newFlags) {
                delete[] newTable;
                return false;
            }
            std::fill_n(newFlags, flagBytes, static_cast<uint8_t>(0));

            T* oldTable = table;
            uint8_t* oldFlags = flags;
            uint8_t oldCap = cap_;

            table = newTable;
            flags = newFlags;
            cap_ = newCap;
            size_ = 0;
            dead_size_ = 0;
            virtual_cap = cap_to_virtual();
            step_ = calStep(newCap);

            if (oldTable && oldFlags) {
                for (uint8_t i = 0; i < oldCap; ++i) {
                    if (getStateFrom(oldFlags, i) == slotState::Used) {
                        insert(oldTable[i]);
                    }
                }
            }

            if (oldFlags) {
                delete[] oldFlags;
            }
            delete[] oldTable;
            return true;
        }
        // safely convert between cap_ and virtual_cap 
        // ensure integrity after 2-way conversion
        uint8_t cap_to_virtual() const noexcept {
            return static_cast<uint8_t>((static_cast<uint16_t>(cap_) * fullness_) / 100);
        }
        
        uint8_t virtual_to_cap(uint8_t v_cap) const noexcept {
            return static_cast<uint8_t>((static_cast<uint16_t>(v_cap) * 100) / fullness_);
        }
        bool is_full() const noexcept {
            return size_ >= virtual_cap;
        }
    public:
        /**
         * @brief Default constructor, creates an unordered_set_s with small initial capacity.
         */
        unordered_set_s() noexcept {
            rehash(4);
        }

        /**
         * @brief Constructor with specified initial capacity.
         * @param cap Initial capacity (number of elements) the set should accommodate.
         */
        explicit unordered_set_s(uint8_t cap) noexcept {
            rehash(cap);
        }

        /**
         * @brief Destructor, frees all allocated memory.
         */
        ~unordered_set_s() noexcept {
            delete[] table;
        }

        /**
         * @brief Copy constructor, creates a deep copy of another set.
         * @param other The set to copy from.
         */
        unordered_set_s(const unordered_set_s& other) noexcept : hash_kernel(),
            slot_handler(other),  // copy flags & cap_
            fullness_(other.fullness_),
            virtual_cap(other.virtual_cap),
            step_(other.step_)
        {
            cap_ = other.cap_;
            size_ = other.size_;
            dead_size_ = other.dead_size_;
            table = new (std::nothrow) T[cap_];
            for (uint8_t i = 0; i < cap_; ++i) {
                if (getState(i) == slotState::Used)
                    table[i] = other.table[i];
                else
                    table[i] = T(); // clear unused slots for safety
            }
        }

        /**
         * @brief Move constructor, transfers ownership of resources.
         * @param other The set to move from (will be left in a valid but unspecified state).
         */
        unordered_set_s(unordered_set_s&& other) noexcept : hash_kernel(),
        slot_handler(std::move(other)),  // ← steal flags & cap_,
        size_(other.size_),
        dead_size_(other.dead_size_),
        fullness_(other.fullness_),
        virtual_cap(other.virtual_cap),
        step_(other.step_){
            table       = other.table;
            other.table = nullptr;
            other.size_ = 0;
            other.fullness_ = 92;
            other.virtual_cap = 0;
            other.step_ = 0;
            other.dead_size_ = 0;
        }

        /**
         * @brief Copy assignment operator, replaces contents with a copy of another set.
         * @param other The set to copy from.
         * @return Reference to *this.
         */
        unordered_set_s& operator=(const unordered_set_s& other) noexcept {
            if (this != &other) {
                delete[] table;
                slot_handler::operator=(other);  // copy flags & cap_
                fullness_ = other.fullness_;
                virtual_cap = other.virtual_cap;
                step_ = other.step_;
                cap_ = other.cap_;
                size_ = other.size_;
                dead_size_ = other.dead_size_;
                table = new (std::nothrow) T[cap_];
                for (uint8_t i = 0; i < cap_; ++i) {
                    if (getState(i) == slotState::Used)
                        table[i] = other.table[i];
                }
            }
            return *this;
        }

        /**
         * @brief Move assignment operator, transfers ownership of resources.
         * @param other The set to move from (will be left in a valid but unspecified state).
         * @return Reference to *this.
         */
        unordered_set_s& operator=(unordered_set_s&& other) noexcept {
            if (this != &other) {
                delete[] table;
                slot_handler::operator=(std::move(other)); // steal flags & cap_
                size_      = other.size_;
                dead_size_ = other.dead_size_;
                fullness_  = other.fullness_;
                table      = other.table;
                virtual_cap = other.virtual_cap;
                step_      = other.step_;
                // reset other
                other.table = nullptr;
                other.size_ = 0;
                other.dead_size_ = 0;
                other.cap_  = 0;
                other.fullness_ = 92;
                other.virtual_cap = 0;
                other.step_ = 0;
            }
            return *this;
        }

        template<bool IsConst>
        class base_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = typename std::conditional<IsConst, const T*, T*>::type;
            using reference         = typename std::conditional<IsConst, const T&, T&>::type;

        private:
            typename std::conditional<IsConst, const unordered_set_s*, unordered_set_s*>::type set_;
            uint8_t index_;

            void findNextUsed() {
                while (index_ < set_->cap_ && set_->getState(index_) != slotState::Used) {
                    ++index_;
                }
            }
            void findPrevUsed() {
                if (index_ == 0) return;
                uint8_t i = index_ - 1;
                while (i > 0 && set_->getState(i) != slotState::Used) --i;
                if (set_->getState(i) == slotState::Used) index_ = i;
            }

        public:
            // default constructor for "end()"
            base_iterator() noexcept
                : set_(nullptr), index_(MAX_CAP)
            {}
            base_iterator(typename std::conditional<IsConst, const unordered_set_s*, unordered_set_s*>::type set,
                        uint8_t start)
            : set_(set), index_(start) {
                findNextUsed();
            }

            base_iterator& operator++() {
                if (index_ < set_->cap_) {
                    ++index_;
                    findNextUsed();
                }
                return *this;
            }
            base_iterator operator++(int) {
                base_iterator tmp = *this;
                ++(*this);
                return tmp;
            }
            base_iterator& operator--() {
                findPrevUsed();
                return *this;
            }
            base_iterator operator--(int) {
                base_iterator tmp = *this;
                --(*this);
                return tmp;
            }

            reference operator*()  const { return set_->table[index_]; }
            pointer   operator->() const { return &set_->table[index_]; }

            bool operator==(const base_iterator& o) const { return index_ == o.index_; }
            bool operator!=(const base_iterator& o) const { return !(*this == o); }
        };

        using iterator       = base_iterator<false>;
        using const_iterator = base_iterator<true>;
        
        // Iterators set     
        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, cap_); }
        const_iterator begin() const { return const_iterator(this, 0); }
        const_iterator end() const { return const_iterator(this, cap_); }
        const_iterator cbegin() const { return const_iterator(this, 0); }
        const_iterator cend() const { return const_iterator(this, cap_); }

        /**
         * @brief Inserts an element into the set, with perfect forwarding.
         * @param value The value to insert.
         * @return true if insertion took place, false if the element already exists.
         * @note Uses perfect forwarding to minimize copies and handle both lvalues and rvalues.
         */
        template<typename U>
        bool insert(U&& value) noexcept {
            if (dead_size_ >= virtual_cap) {
                if (size_ == set_ability())
                    return false;
                uint16_t dbl = cap_ ? cap_ * 2: INIT_CAP;
                if (dbl > MAX_CAP) dbl = MAX_CAP;
                if (!rehash(static_cast<uint8_t>(dbl))) {
                    return false;
                }
            }

            if (cap_ == 0 || table == nullptr) {
                return false;
            }
            
            uint8_t index = hashFunction(cap_, value, hashers[cap_ - 1]);
            
            while (getState(index) != slotState::Empty) {
                auto st = getState(index);
                if(table[index] == value){
                    if(st == slotState::Used){
                        return false;       // duplicate
                    } 
                    if(st == slotState::Deleted){
                        break;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            slotState oldState = getState(index);
            if(oldState == slotState::Empty){
                ++dead_size_;
            }
            table[index] = std::forward<U>(value);
            setState(index, slotState::Used);
            ++size_;
            return true;
        }

        /**
         * @brief Removes an element with the specified value.
         * @param value The value to find and remove.
         * @return true if an element was removed, false otherwise.
         */
        bool erase(const T& value) noexcept {
            if (cap_ == 0 || table == nullptr) {
                return false;
            }
            uint8_t index = hashFunction(cap_, value, hashers[cap_ - 1]);
            uint8_t attempt = 0;
            while(getState(index) != slotState::Empty){
                if(attempt++ == cap_){
                    return false;
                }
                if(table[index] == value){
                    if(getState(index) == slotState::Used){
                        setState(index, slotState::Deleted);
                        --size_;
                        return true;
                    }
                    else if(getState(index) == slotState::Deleted){
                        return false;
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return false;
        }

        /**
         * @brief Finds an element with the specified value.
         * @param value The value to find.
         * @return Iterator to the element if found, otherwise end().
         */
        iterator find(const T& value) noexcept {
            if (cap_ == 0 || table == nullptr) {
                return end();
            }
            uint8_t index = hashFunction(cap_, value, hashers[cap_ - 1]);
            uint8_t attempt = 0;
            while(getState(index) != slotState::Empty){
                if(attempt++ >= cap_){
                    return end();
                }
                if(table[index] == value){
                    if(getState(index) == slotState::Used){
                        return iterator(this,index);
                    }
                    else if(getState(index) == slotState::Deleted){
                        return end();
                    }
                }
                index = linearShifting(cap_, index, step_);
            }
            return end();
        }

        /**
         * @brief Checks if this set contains the same elements as another.
         * @param other The set to compare with.
         * @return true if sets are equal, false otherwise.
         */
        bool operator==(const unordered_set_s& other) const noexcept {
            if (size_ != other.size_) return false;
            for (uint8_t i = 0; i < cap_; ++i) {
                if (getState(i) == slotState::Used && !other.contains(table[i])) {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Checks if this set differs from another.
         * @param other The set to compare with.
         * @return true if sets are not equal, false otherwise.
         */
        bool operator!=(const unordered_set_s& other) const noexcept {
            return !(*this == other);
        }

        /**
         * @brief Gets the current fullness factor.
         * @return The current fullness factor as a float (0.0 to 1.0).
         */
        float get_fullness() const noexcept {
            return static_cast<float>(fullness_) / 100.0f;
        }

        /**
         * @brief Sets the fullness factor for the set.
         * @param fullness The new fullness factor (0.1 to 1.0 or 10 to 100).
         * @return true if successful, false if the new fullness would overflow the set.
         * @note Lower fullness reduces collisions but increases memory usage:
         *       0.9 -> -71% collisions | +11% memory
         *       0.8 -> -87% collisions | +25% memory
         *       0.7 -> -94% collisions | +43% memory
         */
        bool set_fullness(float fullness) noexcept {
            // Ensure fullness is within the valid range [0.1, 1.0] or [10, 100] (% format)
            if(fullness < 0.1f) fullness = 0.1f;
            if(fullness > 1.0f && fullness < 10) fullness = 1.0f;
            if(fullness > 100) fullness = 100;

            uint8_t old_fullness_ = fullness_;

            if(fullness <= 1.0f){
                // Convert fullness to an integer percentage
                uint8_t newFullness = static_cast<uint8_t>(fullness * 100);
                fullness_ = newFullness;
            }else{
                fullness_ = static_cast<uint8_t>(fullness);
            }

            if(set_ability() < size_){
                fullness_ = old_fullness_;
                return false;
            }
            return true;
        }

        /**
         * @brief Checks if the set contains a specific value.
         * @param value The value to find.
         * @return true if the value exists in the set, false otherwise.
         */
        bool contains(const T& value) noexcept {
            if(find(value) == end()) return false;
            return true;
        }

        /**
         * @brief Shrinks the set's capacity to fit its size.
         * @return Number of bytes freed by shrinking.
         */
        size_t shrink_to_fit() noexcept {
            if (size_ < cap_) {
                const uint8_t oldCap = cap_;
                const size_t oldFlagBytes = (static_cast<size_t>(oldCap) * 2 + 7) / 8;
                if (!rehash(size_)) {
                    return 0;
                }
                const size_t newFlagBytes = (static_cast<size_t>(cap_) * 2 + 7) / 8;
                const size_t tableSaved = (oldCap > cap_) ?
                    static_cast<size_t>(oldCap - cap_) * sizeof(T) : 0;
                const size_t flagsSaved = (oldFlagBytes > newFlagBytes) ?
                    oldFlagBytes - newFlagBytes : 0;
                return tableSaved + flagsSaved;
            }
            return 0;
        }

        /**
         * @brief Resizes the set to a new capacity.
         * @param new_virtual_cap The new virtual capacity.
         * @return true if successful, false if the requested capacity is too large.
         */
        bool resize(uint8_t new_virtual_cap) noexcept {
            uint8_t newCap = virtual_to_cap(new_virtual_cap);
            if (newCap > MAX_CAP) return false;
            if (newCap < size_) newCap = size_;
            if (newCap == cap_) return true;
            return rehash(newCap);
        }

        /**
         * @brief Reserves space for a specified number of elements.
         * @param virtual_cap The new virtual capacity to reserve.
         * @return true if successful, false if the requested capacity is too large.
         * @note This prepares the set to hold the specified number of elements without rehashing.
         */
        bool reserve(uint8_t virtual_cap) noexcept {
            uint8_t newCap = virtual_to_cap(virtual_cap);
            if (newCap > MAX_CAP) return false;
            if (newCap < size_) newCap = size_;
            if (newCap == cap_) return true;
            return rehash(newCap);
        }

        /**
         * @brief Gets the maximum theoretical number of elements the set can hold.
         * @return Maximum capacity based on the current fullness setting.
         */
        uint16_t set_ability() const noexcept {
            return static_cast<uint16_t>(MAX_CAP) * fullness_ / 100;
        }

        /**
         * @brief Gets the current number of elements in the set.
         * @return The element count.
         */
        uint16_t size() const noexcept {
            return size_;
        }

        /**
         * @brief Gets the current virtual capacity of the set.
         * @return The current virtual capacity.
         */
        uint16_t capacity() const noexcept {
            return virtual_cap;
        }

        /**
         * @brief Checks if the set is empty.
         * @return true if the set contains no elements, false otherwise.
         */
        bool empty() const noexcept {
            return size_ == 0;
        }

        /**
         * @brief Removes all elements from the set.
         * @note Keeps the allocated memory for reuse.
         */
        void clear() noexcept {
            if (!flags) {
                size_ = 0;
                dead_size_ = 0;
                return;
            }
            std::fill_n(flags, (static_cast<size_t>(cap_) * 2 + 7) / 8, static_cast<uint8_t>(0));
            size_ = 0;
            dead_size_ = 0;
        }

        /**
         * @brief Calculates the total memory usage of the set.
         * @return Memory usage in bytes.
         * @note Includes the set object, elements table, and flag array.
         */
        size_t memory_usage() const noexcept {
            size_t table_bytes = static_cast<size_t>(cap_) * sizeof(T);
            size_t flags_bytes = (cap_ * 2 + 7) / 8;
            return sizeof(*this) + table_bytes + flags_bytes;
        }

        /**
         * @brief Checks if the table pointer is allocated in PSRAM.
         * @return true if table is in PSRAM, false if in DRAM or null.
         */
        bool is_table_in_psram() const noexcept {
            return false;
        }

        /**
         * @brief Gets the table pointer for debugging (returns nullptr if not allocated).
         * @return Pointer to the internal table (may be in PSRAM or DRAM).
         */
        const T* get_table_ptr() const noexcept {
            return table;
        }

        /**
         * @brief Swaps the contents of two sets.
         * @param other The set to swap with.
         */
        void swap(unordered_set_s& other) noexcept {
            std::swap(table, other.table);
            std::swap(flags, other.flags);
            std::swap(cap_, other.cap_);
            std::swap(size_, other.size_);
            std::swap(dead_size_, other.dead_size_);
            std::swap(fullness_, other.fullness_);
            std::swap(virtual_cap, other.virtual_cap);
            std::swap(step_, other.step_);
        }
    };

    

/*
    ------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------- VECTOR ---------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
*/

    template<typename T>
    class vector : hash_kernel{
    private:
        T*      array    = nullptr;
        size_t size_    = 0;
        size_t capacity_ = 0;
        
        // Maximum capacity for the vector
        static constexpr size_t VECTOR_MAX_CAP = 2000000000;

        // internal resize without preserving size
        void i_resize(size_t newCapacity) noexcept {
            if (newCapacity == capacity_) return;
            if (newCapacity == 0) newCapacity = 1;
            T* newArray = new (std::nothrow) T[newCapacity];
            size_t toCopy = (size_ < newCapacity ? size_ : newCapacity);
            customCopy(array, newArray, toCopy);
            delete[] array;
            array = newArray;
            capacity_ = newCapacity;
            if (size_ > capacity_) size_ = capacity_;
        }

        // Internal helper: copy 'count' elements from src to dst
        void customCopy(const T* src, T* dst, size_t count) noexcept {
            if (!src || !dst || count == 0) {
                return;
            }
            if constexpr (std::is_trivially_copyable_v<T>) {
                const auto bytes = count * sizeof(T);
                const auto* src_bytes = reinterpret_cast<const uint8_t*>(src);
                auto* dst_bytes = reinterpret_cast<uint8_t*>(dst);
                if (dst_bytes + bytes <= src_bytes || src_bytes + bytes <= dst_bytes) {
                    std::memcpy(dst, src, bytes);
                } else {
                    std::memmove(dst, src, bytes);
                }
            } else {
                for (size_t i = 0; i < count; ++i) {
                    dst[i] = src[i];
                }
            }
        }

    public:
        // Default: allocate capacity=1
        vector() noexcept
            : array(new (std::nothrow) T[1]), size_(0), capacity_(1) {}

        // Constructor with initial capacity
        explicit vector(size_t initialCapacity) noexcept
            : array(new (std::nothrow) T[(initialCapacity == 0) ? 1 : initialCapacity]),
            size_(initialCapacity), // Set size equal to capacity
            capacity_((initialCapacity == 0) ? 1 : initialCapacity) {
            // fix for error : access elements throgh operator[] when vector just initialized
            for (size_t i = 0; i < size_; ++i) {
                array[i] = T();
            }
        }

        // Constructor with initial size and value
        explicit vector(size_t initialCapacity, const T& value) noexcept
            : array(new (std::nothrow) T[(initialCapacity == 0) ? 1 : initialCapacity]),
            size_(initialCapacity), capacity_((initialCapacity == 0) ? 1 : initialCapacity) {
            for (size_t i = 0; i < size_; ++i) {
                array[i] = value;
            }
        }

        // Constructor: from min_init_list<T>
        vector(const mcu::min_init_list<T>& init) noexcept
            : array(new (std::nothrow) T[init.size()]), size_(init.size()), capacity_(init.size()) {
            for (unsigned i = 0; i < init.size(); ++i)
                array[i] = init.data_[i];
        }

        // Copy constructor
        vector(const vector& other) noexcept
            : array(new (std::nothrow) T[other.size_]),
            size_(other.size_), capacity_(other.size_) {
            if (size_ > 0) {
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(array, other.array, size_ * sizeof(T));
                } else {
                    for (size_t i = 0; i < size_; ++i) {
                        array[i] = other.array[i];
                    }
                }
            }
        }

        // Move constructor
        vector(vector&& other) noexcept
            : array(other.array), size_(other.size_), capacity_(other.capacity_) {
            other.array    = nullptr;
            other.size_    = 0;
            other.capacity_ = 0;
        }

        // Destructor
        ~vector() noexcept {
            delete[] array;
        }

        // Copy assignment
        vector& operator=(const vector& other) noexcept {
            if (this != &other) {
                // Reuse existing storage when possible.
                if (other.size_ <= capacity_ && array != nullptr) {
                    if constexpr (std::is_trivially_copyable_v<T>) {
                        std::memcpy(array, other.array, other.size_ * sizeof(T));
                    } else {
                        for (size_t i = 0; i < other.size_; ++i) {
                            array[i] = other.array[i];
                        }
                    }
                    size_ = other.size_;
                } else {
                    const size_t newCap = (other.size_ == 0) ? 1 : other.size_;
                    T* newArray = new (std::nothrow) T[newCap];
                    if (other.size_ > 0) {
                        if constexpr (std::is_trivially_copyable_v<T>) {
                            std::memcpy(newArray, other.array, other.size_ * sizeof(T));
                        } else {
                            for (size_t i = 0; i < other.size_; ++i) {
                                newArray[i] = other.array[i];
                            }
                        }
                    }
                    delete[] array;
                    array = newArray;
                    size_ = other.size_;
                    capacity_ = newCap;
                }
            }
            return *this;
        }

        // Move assignment
        vector& operator=(vector&& other) noexcept {
            if (this != &other) {
                delete[] array;
                array      = other.array;
                size_      = other.size_;
                capacity_  = other.capacity_;
                other.array    = nullptr;
                other.size_    = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        // Reserve at least newCapacity
        void reserve(size_t newCapacity) noexcept {
            if (newCapacity > capacity_) i_resize(newCapacity);
        }

        // Append element
        void push_back(const T& value) noexcept {
            if (size_ == capacity_) {
                size_t doubled;
                if(VECTOR_MAX_CAP == 255)
                    doubled = capacity_ ? capacity_ + 10 : 1;
                else
                    doubled = capacity_ ? capacity_ * 2 : 1;
                if (doubled > VECTOR_MAX_CAP) doubled = VECTOR_MAX_CAP;
                i_resize(doubled);
            }
            array[size_++] = value;
        }

        // Construct element in place at the end
        template<typename... Args>
        void emplace_back(Args&&... args) noexcept {
            if (size_ == capacity_) {
                size_t doubled;
                if(VECTOR_MAX_CAP == 255)
                    doubled = capacity_ ? capacity_ + 10 : 1;
                else
                    doubled = capacity_ ? capacity_ * 20 : 1;
                if (doubled > VECTOR_MAX_CAP) doubled = VECTOR_MAX_CAP;
                i_resize(doubled);
            }
            new (&array[size_++]) T(std::forward<Args>(args)...);
        }

        // Insert at position
        void insert(size_t pos, const T& value) noexcept {
            if (pos > size_) return;
            if (size_ == capacity_) {
                size_t doubled;
                if(VECTOR_MAX_CAP == 255)
                    doubled = capacity_ ? capacity_ + 10 : 1;
                else
                    doubled = capacity_ ? capacity_ * 20 : 1;
                if (doubled > VECTOR_MAX_CAP) doubled = VECTOR_MAX_CAP;
                i_resize(doubled);
            }
            for (size_t i = size_; i > pos; --i) {
                array[i] = array[i - 1];
            }
            array[pos] = value;
            ++size_;
        }

        // Construct element in place at position
        template<typename... Args>
        void emplace(size_t pos, Args&&... args) noexcept {
            if (pos > size_) return;
            if (size_ == capacity_) {
                size_t doubled;
                if(VECTOR_MAX_CAP == 255)
                    doubled = capacity_ ? capacity_ + 10 : 1;
                else
                    doubled = capacity_ ? capacity_ * 20 : 1;
                if (doubled > VECTOR_MAX_CAP) doubled = VECTOR_MAX_CAP;
                i_resize(doubled);
            }
            for (size_t i = size_; i > pos; --i) {
                array[i] = array[i - 1];
            }
            new (&array[pos]) T(std::forward<Args>(args)...);
            ++size_;
        }

        /*
        Inserts a range into the %vector.

        Parameters:
        __position – A const_iterator into the %vector.
        __first – An input iterator.
        __last – An input iterator.
        */
        template<typename InputIterator>
        void insert(const T* position, InputIterator first, InputIterator last) noexcept {
            size_t pos = position - array;
            size_t count = last - first;
            if (pos > size_) return;
            if (size_ + count > capacity_) {
                size_t newCapacity = capacity_ ? capacity_ * 2 : 1;
                if (newCapacity > VECTOR_MAX_CAP) newCapacity = VECTOR_MAX_CAP;
                i_resize(newCapacity);
            }
            for (size_t i = size_ + count - 1; i >= pos + count; --i) {
                array[i] = array[i - count];
            }
            for (size_t i = 0; i < count; ++i) {
                array[pos + i] = *(first + i);
            }
            size_ += count;
        }

        void sort() noexcept {
            static_assert(std::is_arithmetic<T>::value || less_comparable<T>::value,
                          "Type T must be numeric or support operator< (returning convertible-to-bool) for sorting.");
            // Safety check: null array pointer
            if (array == nullptr) return;
            
            // Safety check: basic size validation
            if (size_ <= 1) return;
            
            // Safety check: size consistency
            if (size_ > capacity_) {
                size_ = capacity_; // Fix corrupted size
            }
            
            // Safety check: prevent integer overflow on large arrays
            if (size_ >= VECTOR_MAX_CAP) return;
            
            quickSort(0, size_ - 1);
        }
    private:
        // Helper function to compare two elements, using preprocessing for non-numeric types
        bool is_less(const T& a, const T& b) noexcept {
            if constexpr (std::is_arithmetic_v<T>) {
                // For numeric types, direct comparison
                return a < b;
            } else {
                // For non-numeric types, use hash preprocessing
                size_t hash_a = this->preprocess_hash_input(a);
                size_t hash_b = this->preprocess_hash_input(b);
                return hash_a < hash_b;
            }
        }

        // Partition function with comprehensive safety checks
        size_t partition(size_t low, size_t high) noexcept {
            // Safety: null pointer check
            if (array == nullptr) return low;
            
            // Safety: boundary validation
            if (low >= size_ || high >= size_) return low;
            if (low > high) return low;
            
            // Safety: handle unsigned underflow case
            if (high == 0 && low > 0) return low;
            
            T pivot = array[high];
            size_t i = low;
            
            // Safety: bound-checked loop
            for (size_t j = low; j < high && j < size_; ++j) {
                // Additional bounds check during iteration
                if (i >= size_) break;
                
                if (is_less(array[j], pivot)) {
                    // Safety: validate both indices before swap
                    if (i < size_ && j < size_) {
                        T temp = array[i];
                        array[i] = array[j];
                        array[j] = temp;
                    }
                    ++i;
                    
                    // Safety: prevent index overflow
                    if (i >= size_) break;
                }
            }
            
            // Safety: final bounds check before pivot placement
            if (i < size_ && high < size_) {
                T temp = array[i];
                array[i] = array[high];
                array[high] = temp;
            }
            
            return i;
        }
        
        // Quicksort with stack overflow protection and safety checks
        void quickSort(size_t low, size_t high) noexcept {
            // Safety: null pointer check
            if (array == nullptr) return;
            
            // Safety: boundary validation
            if (low >= size_ || high >= size_) return;
            if (low >= high) return;
            
            // Safety: stack overflow protection
            static uint8_t recursion_depth = 0;
            const uint8_t MAX_RECURSION_DEPTH = 24; // Conservative limit for embedded
            
            if (recursion_depth >= MAX_RECURSION_DEPTH) {
                // Fall back to iterative bubble sort for safety
                bubbleSortFallback(low, high);
                return;
            }
            
            // Safety: detect infinite recursion patterns
            if (high - low > size_) return; // Invalid range
            
            ++recursion_depth;
            
            size_t pivotIndex = partition(low, high);
            
            // Safety: validate pivot index before recursive calls
            if (pivotIndex >= low && pivotIndex <= high && pivotIndex < size_) {
                // Sort left partition with underflow protection
                if (pivotIndex > low && pivotIndex > 0) {
                    quickSort(low, pivotIndex - 1);
                }
                // Sort right partition
                if (pivotIndex < high && pivotIndex + 1 < size_) {
                    quickSort(pivotIndex + 1, high);
                }
            }
            
            --recursion_depth;
        }
        
        // Safe fallback sorting when recursion limit reached
        void bubbleSortFallback(size_t low, size_t high) noexcept {
            // Safety: basic validation
            if (array == nullptr || low >= high || high >= size_) return;
            
            // Safety: prevent infinite loops
            const size_t max_iterations = (high - low + 1) * (high - low + 1);
            size_t iteration_count = 0;
            
            for (size_t i = low; i <= high && i < size_; ++i) {
                for (size_t j = low; j < high - (i - low) && j < size_; ++j) {
                    // Safety: iteration limit check
                    if (++iteration_count > max_iterations) return;
                    
                    // Safety: bounds check for each access
                    if (j + 1 <= high && j < size_ && j + 1 < size_) {
                        if (!is_less(array[j], array[j + 1]) && !is_less(array[j + 1], array[j])) {
                            // Elements are equal, no swap needed
                        } else if (!is_less(array[j], array[j + 1])) {
                            T temp = array[j];
                            array[j] = array[j + 1];
                            array[j + 1] = temp;
                        }
                    }
                }
            }
        }

    public:
        // Erase element at position
        void erase(size_t pos) noexcept {
            if (pos >= size_) return;
            customCopy(array + pos + 1, array + pos, size_ - pos - 1);
            --size_;
        }

        /**
         * @brief Removes a single element at the given iterator position.
         * @param position Iterator pointing to the element to be removed.
         * @return Iterator pointing to the element that followed the erased element, or end() if the last element was erased.
         */
        T* erase(const T* position) noexcept {
            if (position < begin() || position >= end()) {
                return end();
            }
            size_t idx = static_cast<size_t>(position - begin());
            erase(idx);
            return begin() + idx;
        }

        /**
         * @brief Removes a range of elements [first, last).
         * @param first Iterator to the first element to be removed.
         * @param last Iterator to one past the last element to be removed.
         * @return Iterator pointing to the element that followed the last erased element, or end() if all elements to the end were erased.
         */
        T* erase(const T* first, const T* last) noexcept {
            if (first < begin()) {
                first = begin();
            }
            if (last > end()) {
                last = end();
            }
            if (first >= last) {
                return begin() + static_cast<size_t>(first - begin());
            }

            size_t start = static_cast<size_t>(first - begin());
            size_t stop = static_cast<size_t>(last - begin());
            if (stop > size_) stop = size_;
            size_t count = stop - start;
            if (count == 0) {
                return begin() + start;
            }

            customCopy(array + stop, array + start, size_ - stop);
            size_ -= count;
            return begin() + start;
        }

        bool empty() const noexcept {
            return size_ == 0;
        }

        // Clear contents (keep capacity)
        void clear() noexcept {
            size_ = 0;
        }

        void fill(const T& value) noexcept {
            for (size_t i = 0; i < size_; ++i) {
                array[i] = value;
            }
            size_ = capacity_;
        }

        // Shrink capacity to fit size
        void shrink_to_fit() noexcept {
            if (size_ < capacity_) i_resize(size_);
        }
        T& back() noexcept {
            if (size_ == 0){
                static T default_value = T();
                return default_value; // Return default value if empty
            }
            return array[size_ - 1];
        }
        const T& back() const noexcept {
            if (size_ == 0){
                static T default_value = T();
                return default_value; // Return default value if empty
            }
            return array[size_ - 1];
        }
        T& front() noexcept {
            if (size_ == 0){
                static T default_value = T();
                return default_value; // Return default value if empty
            }
            return array[0];
        }
        const T& front() const noexcept {
            if (size_ == 0){
                static T default_value = T();
                return default_value; // Return default value if empty
            }
            return array[0];
        }

        // emplace_back: construct an element in place at the end
        
        void pop_back() noexcept {
            if (size_ == 0) {
                return; // Do nothing if empty
            }
            --size_;
        }
        // Returns a pointer such that [data(), data() + size()) is a valid range. For a non-empty %vector, data() == &front().
        T* data() noexcept { return array; }
        const T* data() const noexcept { return array; }

        /**
         * Resizes the container so that it contains n elements.

        - If n is smaller than the current container size, the content is reduced to its first n elements, removing those beyond (and destroying them).

        - If n is greater than the current container size, the content is expanded by inserting at the end as many elements as needed to reach a size of n. If val is specified, the new elements are initialized as copies of val, otherwise, they are value-initialized.

        - If n is also greater than the current container capacity, an automatic reallocation of the allocated storage space takes place.
         */
        void resize(size_t newSize, const T& value) noexcept {
            if (newSize > VECTOR_MAX_CAP) {
                newSize = VECTOR_MAX_CAP;
            }
            
            if (newSize < size_) {
                // Shrink: reduce size to newSize
                size_ = newSize;
            } else if (newSize > size_) {
                // Expand: need to add elements
                if (newSize > capacity_) {
                    // Need to reallocate
                    i_resize(newSize);
                }
                
                // Fill new elements with the provided value
                for (size_t i = size_; i < newSize; ++i) {
                    array[i] = value;
                }
                size_ = newSize;
            }
            // If newSize == size_, do nothing
        }
        
        void resize(size_t newSize) noexcept {
            if (newSize > VECTOR_MAX_CAP) {
                newSize = VECTOR_MAX_CAP;
            }
            
            if (newSize < size_) {
                // Shrink: reduce size to newSize
                size_ = newSize;
            } else if (newSize > size_) {
                // Expand: need to add elements
                if (newSize > capacity_) {
                    // Need to reallocate
                    i_resize(newSize);
                }
                
                // Fill new elements with default-initialized values
                for (size_t i = size_; i < newSize; ++i) {
                    array[i] = T();
                }
                size_ = newSize;
            }
            // If newSize == size_, do nothing
        }

        void assign(size_t count, const T& value) noexcept {
            if (count > VECTOR_MAX_CAP) {
                count = VECTOR_MAX_CAP;
            }
            clear();
            reserve(count);
            resize(count, value);
        }

        // Accessors
        size_t size() const noexcept { return size_; }
        size_t capacity() const noexcept { return capacity_; }

        // Operator[] returns default T() on out-of-range
        T& operator[](size_t index) noexcept {
            static T default_value = T();
            return (index < size_) ? array[index] : default_value;
        }
    
        const T& operator[](size_t index) const noexcept {
            static T default_value = T();
            return (index < size_) ? array[index] : default_value;
        }

        // Iterators (raw pointers)
        T* begin() noexcept { return array; }
        T* end()   noexcept { return array + size_; }
        const T* begin() const noexcept { return array; }
        const T* end()   const noexcept { return array + size_; }

        size_t memory_usage() const noexcept {
            size_t heap_bytes = static_cast<size_t>(capacity_) * sizeof(T);
            return sizeof(*this) + heap_bytes;
        }
    };
    
    /*
    -------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------- PACKED VECTOR ---------------------------------------------------
    -------------------------------------------------------------------------------------------------------------------
    */
    template<typename>
    struct dependent_false : std::false_type {};

    template<typename T, typename Enable = void>
    struct packed_value_traits {
        static size_t to_bits(const T&) {
            static_assert(dependent_false<T>::value, "packed_value_traits specialization required for this type");
            return 0;
        }

        static T from_bits(size_t) {
            static_assert(dependent_false<T>::value, "packed_value_traits specialization required for this type");
            return T{};
        }
    };

    template<typename T>
    struct packed_value_traits<T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>> {
        static constexpr size_t to_bits(T value) noexcept {
            return static_cast<size_t>(value);
        }

        static constexpr T from_bits(size_t bits) noexcept {
            return static_cast<T>(bits);
        }
    };

    template<typename T>
    struct packed_value_traits<T, std::enable_if_t<!std::is_integral_v<T> && !std::is_enum_v<T> &&
                                                  std::is_trivially_copyable_v<T> &&
                                                  (sizeof(T) <= sizeof(size_t))>> {
        static size_t to_bits(const T& value) noexcept {
            size_t bits = 0;
            const auto* src = reinterpret_cast<const uint8_t*>(&value);
            auto* dst = reinterpret_cast<uint8_t*>(&bits);
            for (size_t i = 0; i < sizeof(T); ++i) {
                dst[i] = src[i];
            }
            return bits;
        }

        static T from_bits(size_t bits) noexcept {
            T value{};
            const auto* src = reinterpret_cast<const uint8_t*>(&bits);
            auto* dst = reinterpret_cast<uint8_t*>(&value);
            for (size_t i = 0; i < sizeof(T); ++i) {
                dst[i] = src[i];
            }
            return value;
        }
    };


    template<uint8_t BitsPerElement>
    class PackedArray {
        static_assert(BitsPerElement > 0, "Invalid bit size");

    public:
        using word_t = size_t;
        static constexpr size_t WORD_BITS = sizeof(word_t) * 8;
    private:
        word_t* data = nullptr;
        uint8_t bpv_ = BitsPerElement;
        size_t capacity_words_ = 0;

    public:
        PackedArray() = default;

        explicit PackedArray(size_t capacity_words)
            : bpv_(BitsPerElement), capacity_words_(capacity_words) {
            if (capacity_words_ > 0) {
                data = new (std::nothrow) word_t[capacity_words_];
                if (data) {
                    std::fill_n(data, capacity_words_, static_cast<word_t>(0));
                } else {
                    capacity_words_ = 0;
                }
            }
        }

        ~PackedArray() {
            delete[] data;
            data = nullptr;
            capacity_words_ = 0;
        }

        PackedArray(const PackedArray& other, size_t words)
            : bpv_(other.bpv_), capacity_words_(words) {
            if (capacity_words_ > 0) {
                data = new (std::nothrow) word_t[capacity_words_];
                if (data) {
                    if (other.data) {
                        std::copy(other.data, other.data + capacity_words_, data);
                    } else {
                        std::fill_n(data, capacity_words_, static_cast<word_t>(0));
                    }
                } else {
                    capacity_words_ = 0;
                }
            }
        }

        PackedArray(PackedArray&& other) noexcept
            : data(other.data), bpv_(other.bpv_), capacity_words_(other.capacity_words_) {
            other.data = nullptr;
            other.capacity_words_ = 0;
        }

        void copy_from(const PackedArray& other, size_t words) {
            if (this == &other) {
                bpv_ = other.bpv_;
                capacity_words_ = words;
                return;
            }

            delete[] data;
            data = nullptr;
            capacity_words_ = words;

            if (capacity_words_ > 0) {
                data = new (std::nothrow) word_t[capacity_words_];
                if (data) {
                    if (other.data) {
                        std::copy(other.data, other.data + capacity_words_, data);
                    } else {
                        std::fill_n(data, capacity_words_, static_cast<word_t>(0));
                    }
                } else {
                    capacity_words_ = 0;
                }
            }
            bpv_ = other.bpv_;
        }

        PackedArray& operator=(const PackedArray& other) {
            if (this != &other) {
                copy_from(other, other.capacity_words_);
            }
            return *this;
        }

        PackedArray& operator=(PackedArray&& other) noexcept {
            if (this != &other) {
                delete[] data;
                data = other.data;
                bpv_ = other.bpv_;
                capacity_words_ = other.capacity_words_;
                other.data = nullptr;
                other.capacity_words_ = 0;
            }
            return *this;
        }

        uint8_t get_bpv() const { return bpv_; }

        void set_bpv(uint8_t new_bpv) {
            if (new_bpv == 0) {
                return;
            }
            // On 32-bit systems, we can't support > 32 bits per value with size_t storage                                                      
            if (new_bpv > WORD_BITS) {
                return;
            }
            bpv_ = new_bpv;
        }

    __attribute__((always_inline)) inline void set_unsafe(size_t index, size_t value) {
            if (!data) {
                return;
            }

            const uint8_t active_bpv = bpv_;
            // Fast path: if value fits in one word (common case)
            const size_t bitPos = index * static_cast<size_t>(active_bpv);
            const size_t wordIdx = bitPos / WORD_BITS;
            
            if (wordIdx >= capacity_words_) {
                return;
            }

            size_t bitOff = bitPos % WORD_BITS;
            
            // Check if the element is contained within a single word
            if (bitOff + active_bpv <= WORD_BITS) {
                const size_t mask = (active_bpv >= WORD_BITS)
                    ? static_cast<size_t>(std::numeric_limits<size_t>::max())
                    : ((static_cast<size_t>(1) << active_bpv) - 1ull);
                const size_t clamped = static_cast<size_t>(value) & mask;
                
                word_t& w = data[wordIdx];
                w = (w & ~(mask << bitOff)) | (clamped << bitOff);
                return;
            }

            // Slow path: crosses word boundary
            const size_t mask = (active_bpv >= WORD_BITS)
                ? static_cast<size_t>(std::numeric_limits<size_t>::max())
                : ((static_cast<size_t>(1) << active_bpv) - 1ull);
            const size_t clamped = static_cast<size_t>(value) & mask;

            size_t remaining = active_bpv;
            size_t srcShift = 0;
            size_t wIndex = wordIdx;

            while (remaining > 0) {
                if (wIndex >= capacity_words_) return;
                size_t bitsInWord = std::min<size_t>(WORD_BITS - bitOff, remaining);
                const size_t maskPart = (bitsInWord == WORD_BITS) ? std::numeric_limits<size_t>::max() : ((static_cast<size_t>(1) << bitsInWord) - 1u);
                word_t& w = data[wIndex];
                w = (w & ~(maskPart << bitOff)) | (((clamped >> srcShift) & maskPart) << bitOff);
                remaining -= bitsInWord;
                srcShift += bitsInWord;
                bitOff = 0;
                ++wIndex;
            }
        }

    __attribute__((always_inline)) inline size_t get_unsafe(size_t index) const {
            if (!data) {
                return 0;
            }

            const uint8_t active_bpv = bpv_;
            const size_t bitPos = index * static_cast<size_t>(active_bpv);
            const size_t wordIdx = bitPos / WORD_BITS;
            if (wordIdx >= capacity_words_) {
                return 0;
            }

            size_t bitOff = bitPos % WORD_BITS;
            
            // Fast path: if value fits in one word
            if (bitOff + active_bpv <= WORD_BITS) {
                const size_t mask = (active_bpv >= WORD_BITS)
                    ? static_cast<size_t>(std::numeric_limits<size_t>::max())
                    : ((static_cast<size_t>(1) << active_bpv) - 1ull);
                return (data[wordIdx] >> bitOff) & mask;
            }

            // Slow path: crosses word boundary
            size_t remaining = active_bpv;
            size_t dstShift = 0;
            size_t value = 0;
            size_t wIdx = wordIdx;
            while (remaining > 0) {
                if (wIdx >= capacity_words_) return 0;
                size_t bitsInWord = std::min<size_t>(WORD_BITS - bitOff, remaining);
                const size_t maskPart = (bitsInWord == WORD_BITS) ? std::numeric_limits<size_t>::max() : ((static_cast<size_t>(1) << bitsInWord) - 1u);
                const size_t w = data[wIdx];
                size_t part = (w >> bitOff) & maskPart;
                value |= (part << dstShift);
                remaining -= bitsInWord;
                dstShift += bitsInWord;
                bitOff = 0;
                ++wIdx;
            }
            return value;
        }

        void copy_elements(const PackedArray& src, size_t element_count) {
            if (!data || !src.data) {
                return;
            }

            // Optimization: if bpv matches, we can copy words directly
            if (bpv_ == src.bpv_) {
                const size_t bits_needed = element_count * static_cast<size_t>(bpv_);
                const size_t words_needed = (bits_needed + WORD_BITS - 1) / WORD_BITS;
                const size_t words_to_copy = std::min(words_needed, std::min(capacity_words_, src.capacity_words_));
                
                if (words_to_copy > 0) {
                    std::memcpy(data, src.data, words_to_copy * sizeof(word_t));
                }
                
                // Zero out the remaining words
                for (size_t i = words_to_copy; i < capacity_words_; ++i) {
                    data[i] = 0;
                }
                return;
            }

            for (size_t i = 0; i < element_count; ++i) {
                set_unsafe(i, src.get_unsafe(i));
            }

            const size_t bits_used = element_count * static_cast<size_t>(bpv_);
            const size_t word_bits = WORD_BITS;
            const size_t first_unused_word = (bits_used + word_bits - 1u) / word_bits;
            for (size_t i = first_unused_word; i < capacity_words_; ++i) {
                data[i] = 0;
            }
        }

    void set(size_t index, size_t value) { set_unsafe(index, value); }
    size_t get(size_t index) const { return get_unsafe(index); }

    word_t* raw_data() { return data; }
    const word_t* raw_data() const { return data; }
        size_t words() const { return capacity_words_; }
    };


    template<uint8_t BitsPerElement, typename ValueType = size_t>
    class packed_vector {
    static_assert(BitsPerElement > 0, "Invalid bit size");

    public:
        using value_type = ValueType;
        using size_type = size_t;
        using traits_type = packed_value_traits<value_type>;
        using word_t = typename PackedArray<BitsPerElement>::word_t;

    private:
        PackedArray<BitsPerElement> packed_data;
        size_type size_ = 0;
        size_type capacity_ = 0;

        static constexpr size_type VECTOR_MAX_CAP =
            (std::numeric_limits<size_type>::max() / 2) > 0
                ? (std::numeric_limits<size_type>::max() / 2)
                : static_cast<size_type>(1);

        static constexpr size_t COMPILED_MAX_BITS =
            (BitsPerElement >= (int)PackedArray<BitsPerElement>::WORD_BITS)
                ? std::numeric_limits<size_t>::max()
                : ((static_cast<size_t>(1) << BitsPerElement) - 1u);

        static inline size_t calc_words_for_bpv(size_type capacity, uint8_t bpv) {
            const size_t bits = capacity * static_cast<size_t>(bpv);
            const size_t word_bits = PackedArray<BitsPerElement>::WORD_BITS;
            return (bits + word_bits - 1u) / word_bits;
        }

        static inline size_t runtime_mask(uint8_t bpv) {
            if (bpv >= PackedArray<BitsPerElement>::WORD_BITS) {
                return std::numeric_limits<size_t>::max();
            }
            return (static_cast<size_t>(1) << bpv) - 1u;
        }

        __attribute__((always_inline)) inline size_t mask_bits(size_t bits, uint8_t bpv) const {
            return bits & runtime_mask(bpv);
        }

        __attribute__((always_inline)) inline size_t mask_bits(size_t bits) const {
            return mask_bits(bits, packed_data.get_bpv());
        }

        __attribute__((always_inline)) inline size_t to_storage_bits(const value_type& value, uint8_t bpv) const {
            return mask_bits(traits_type::to_bits(value), bpv);
        }

        __attribute__((always_inline)) inline size_t to_storage_bits(const value_type& value) const {
            return to_storage_bits(value, packed_data.get_bpv());
        }

        __attribute__((always_inline)) inline value_type from_storage_bits(size_t bits, uint8_t bpv) const {
            return traits_type::from_bits(mask_bits(bits, bpv));
        }

        __attribute__((always_inline)) inline value_type from_storage_bits(size_t bits) const {
            return from_storage_bits(bits, packed_data.get_bpv());
        }

        template<typename T>
        struct init_view {
            const T* data;
            size_t count;
        };

        template<typename T>
        static init_view<T> normalize_init_list(mcu::min_init_list<T> init, uint8_t active_bpv) {
            init_view<T> view{init.begin(), static_cast<size_t>(init.size())};
            if (!view.data || view.count == 0) {
                view.count = 0;
                return view;
            }

            bool drop_header = false;
            if (packed_value_traits<T>::to_bits(view.data[0]) == static_cast<size_t>(active_bpv) && view.count > 1) {
                for (size_t i = 1; i < view.count; ++i) {
                    if (packed_value_traits<T>::to_bits(view.data[i]) > static_cast<size_t>(active_bpv)) {
                        drop_header = true;
                        break;
                    }
                }
            }

            if (drop_header) {
                ++view.data;
                --view.count;
            }

            if (view.count > VECTOR_MAX_CAP) {
                view.count = VECTOR_MAX_CAP;
            }

            return view;
        }

        template<typename SourceVector>
        void initialize_from_range(const SourceVector& source, size_t start_index, size_t end_index) {
            uint8_t source_bpv = source.get_bits_per_value();
            uint8_t active_bpv = (source_bpv == 0) ? BitsPerElement : source_bpv;
            if (active_bpv > BitsPerElement) {
                active_bpv = BitsPerElement;
            }

            const size_t source_size = source.size();
            if (start_index > end_index || start_index >= source_size) {
                capacity_ = 1;
                size_ = 0;
                packed_data = PackedArray<BitsPerElement>(calc_words_for_bpv(1, active_bpv));
                packed_data.set_bpv(active_bpv);
                return;
            }

            if (end_index > source_size) {
                end_index = source_size;
            }

            size_ = static_cast<size_type>(end_index - start_index);
            capacity_ = (size_ == 0) ? 1 : size_;

            packed_data = PackedArray<BitsPerElement>(calc_words_for_bpv(capacity_, active_bpv));
            packed_data.set_bpv(active_bpv);

            // Optimized bulk copy for word-aligned ranges when bpv matches
            if (active_bpv == source_bpv && (start_index * active_bpv) % PackedArray<BitsPerElement>::WORD_BITS == 0) {
                // Fast path: word-aligned bulk copy
                const size_t start_bit = start_index * active_bpv;
                const size_t start_word = start_bit / PackedArray<BitsPerElement>::WORD_BITS;
                const size_t num_bits = size_ * active_bpv;
                const size_t num_words = (num_bits + PackedArray<BitsPerElement>::WORD_BITS - 1) / PackedArray<BitsPerElement>::WORD_BITS;
                
                // Direct word copy from source packed data
                const word_t* src_words = source.get_packed_data_words();
                word_t* dst_words = packed_data.raw_data();
                
                if (src_words && dst_words && num_words > 0) {
                    // Use memcpy for bulk transfer
                    memcpy(dst_words, src_words + start_word, num_words * sizeof(word_t));
                    return;
                }
            }
            
            // Fallback: element-by-element copy
            for (size_type i = 0; i < size_; ++i) {
                using SourceValue = typename SourceVector::value_type;
                using SourceTraits = packed_value_traits<SourceValue>;
                const size_t source_bits = SourceTraits::to_bits(source[start_index + i]);
                const value_type converted = traits_type::from_bits(source_bits);
                packed_data.set_unsafe(i, mask_bits(traits_type::to_bits(converted), active_bpv));
            }
        }

        void ensure_capacity(size_type new_capacity) {
            if (new_capacity <= capacity_) {
                return;
            }

            if (new_capacity > VECTOR_MAX_CAP) {
                new_capacity = VECTOR_MAX_CAP;
            }

            const uint8_t active_bpv = packed_data.get_bpv();
            size_type adjusted = (new_capacity == 0) ? 1 : new_capacity;
            size_t words = calc_words_for_bpv(adjusted, active_bpv);
            if (words == 0) {
                words = 1;
            }

            PackedArray<BitsPerElement> new_data(words);
            new_data.set_bpv(active_bpv);
            new_data.copy_elements(packed_data, size_);
            packed_data = std::move(new_data);
            capacity_ = adjusted;
        }

        void init(uint8_t bpv) {
            if (bpv == 0) {
                return;
            }

            size_type target_capacity = (capacity_ == 0) ? 1 : capacity_;
            PackedArray<BitsPerElement> new_data(calc_words_for_bpv(target_capacity, bpv));
            new_data.set_bpv(bpv);
            packed_data = std::move(new_data);
            size_ = 0;
            capacity_ = target_capacity;
        }
    public:
        packed_vector()
            : packed_data(calc_words_for_bpv(1, BitsPerElement)), size_(0), capacity_(1) {}

        explicit packed_vector(size_type initialCapacity)
            : packed_data(calc_words_for_bpv((initialCapacity == 0) ? 1 : initialCapacity, BitsPerElement)),
              size_(0),
              capacity_((initialCapacity == 0) ? 1 : initialCapacity) {}

        packed_vector(size_type initialSize, const value_type& value)
            : packed_data(calc_words_for_bpv((initialSize == 0) ? 1 : initialSize, BitsPerElement)),
              size_(initialSize),
              capacity_((initialSize == 0) ? 1 : initialSize) {
            const uint8_t active_bpv = packed_data.get_bpv();
            const size_t clamped = to_storage_bits(value, active_bpv);
            for (size_type i = 0; i < size_; ++i) {
                packed_data.set_unsafe(i, clamped);
            }
        }

        template<typename T>
        packed_vector(mcu::min_init_list<T> init)
            : packed_vector() {
            assign(init);
        }

        packed_vector(const packed_vector& other)
            : packed_data(other.packed_data, std::max<size_t>(size_t{1}, calc_words_for_bpv(other.capacity_, other.get_bits_per_value()))),
              size_(other.size_),
              capacity_(other.capacity_) {
            packed_data.set_bpv(other.get_bits_per_value());
        }

        packed_vector(packed_vector&& other) noexcept
            : packed_data(std::move(other.packed_data)),
              size_(other.size_),
              capacity_(other.capacity_) {
            other.size_ = 0;
            other.capacity_ = 0;
        }

        packed_vector& operator=(const packed_vector& other) {
            if (this != &other) {
                packed_data.copy_from(other.packed_data, std::max<size_t>(size_t{1}, calc_words_for_bpv(other.capacity_, other.get_bits_per_value())));
                packed_data.set_bpv(other.get_bits_per_value());
                size_ = other.size_;
                capacity_ = other.capacity_;
            }
            return *this;
        }

        packed_vector& operator=(packed_vector&& other) noexcept {
            if (this != &other) {
                packed_data = std::move(other.packed_data);
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.size_ = 0;
                other.capacity_ = 0;
            }
            return *this;
        }

        packed_vector(const packed_vector& source, size_t start_index, size_t end_index) {
            initialize_from_range(source, start_index, end_index);
        }

        template<uint8_t SourceBitsPerElement, typename SourceValue>
        packed_vector(const packed_vector<SourceBitsPerElement, SourceValue>& source, size_t start_index, size_t end_index) {
            initialize_from_range(source, start_index, end_index);
        }

        size_type size() const { return size_; }
        size_type capacity() const { return capacity_; }
        bool empty() const { return size_ == 0; }

        value_type operator[](size_type index) const {
            if (size_ == 0) {
                return value_type{};
            }
            if (index >= size_) {
                return from_storage_bits(packed_data.get_unsafe(size_ - 1));
            }
            return from_storage_bits(packed_data.get_unsafe(index));
        }

        value_type at(size_type index) const {
            if (index >= size_) {
                throw std::out_of_range("packed_vector::at");
            }
            return from_storage_bits(packed_data.get_unsafe(index));
        }

        void set(size_type index, const value_type& value) {
            packed_data.set_unsafe(index, to_storage_bits(value));
        }

        void set_unsafe(size_type index, const value_type& value) {
            packed_data.set_unsafe(index, to_storage_bits(value));
        }

        value_type get(size_type index) const {
            return (index < size_) ? from_storage_bits(packed_data.get_unsafe(index)) : value_type{};
        }

        value_type front() const {
            if (size_ == 0) {
                throw std::out_of_range("packed_vector::front");
            }
            return from_storage_bits(packed_data.get_unsafe(0));
        }

        value_type back() const {
            return (size_ > 0) ? from_storage_bits(packed_data.get_unsafe(size_ - 1)) : value_type{};
        }

        void push_back(const value_type& value) {
            if (size_ == capacity_) {
                size_type new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
                if (new_capacity > VECTOR_MAX_CAP) {
                    new_capacity = VECTOR_MAX_CAP;
                }
                ensure_capacity(new_capacity);
            }
            if (size_ < capacity_) {
                packed_data.set_unsafe(size_, to_storage_bits(value));
                ++size_;
            }
        }

        void pop_back() {
            if (size_ > 0) {
                --size_;
            }
        }

        void fill(const value_type& value) {
            if (size_ == 0) {
                return;
            }
            const uint8_t active_bpv = packed_data.get_bpv();
            const size_t clamped = to_storage_bits(value, active_bpv);

            constexpr size_t word_bits = PackedArray<BitsPerElement>::WORD_BITS;
            if (active_bpv > 0 && (word_bits % active_bpv) == 0) {
                size_t pattern = 0;
                const size_t slots_per_word = word_bits / active_bpv;
                for (size_t slot = 0; slot < slots_per_word; ++slot) {
                    pattern |= (clamped << (slot * active_bpv));
                }

                const size_t total_bits = size_ * static_cast<size_t>(active_bpv);
                const size_t words_to_fill = (total_bits + word_bits - 1) / word_bits;
                if (words_to_fill > 0) {
                    detail_simd::fill_words(packed_data.raw_data(), words_to_fill, pattern);

                    const size_t tail_bits = total_bits % word_bits;
                    if (tail_bits != 0) {
                        const size_t tail_mask = (static_cast<size_t>(1) << tail_bits) - 1u;
                        packed_data.raw_data()[words_to_fill - 1] &= tail_mask;
                    }
                    return;
                }
            }

            for (size_type i = 0; i < size_; ++i) {
                packed_data.set_unsafe(i, clamped);
            }
        }

        void resize(size_type newSize, const value_type& value = value_type{}) {
            if (newSize > capacity_) {
                ensure_capacity(newSize);
            }
            if (newSize > size_) {
                const uint8_t active_bpv = packed_data.get_bpv();
                const size_t clamped = to_storage_bits(value, active_bpv);
                for (size_type i = size_; i < newSize; ++i) {
                    packed_data.set_unsafe(i, clamped);
                }
            }
            size_ = newSize;
        }

        void reserve(size_type newCapacity) {
            ensure_capacity(newCapacity);
        }

        void assign(size_type count, const value_type& value) {
            clear();
            if (count == 0) {
                return;
            }
            ensure_capacity(count);
            const uint8_t active_bpv = packed_data.get_bpv();
            const size_t clamped = to_storage_bits(value, active_bpv);
            for (size_type i = 0; i < count; ++i) {
                packed_data.set_unsafe(i, clamped);
            }
            size_ = count;
        }

        template<typename T>
        void assign(mcu::min_init_list<T> init) {
            auto view = normalize_init_list(init, packed_data.get_bpv());
            clear();
            if (view.count == 0) {
                return;
            }
            ensure_capacity(view.count);
            const uint8_t active_bpv = packed_data.get_bpv();
            for (size_type i = 0; i < view.count; ++i) {
                const size_t bits = packed_value_traits<T>::to_bits(view.data[i]);
                const value_type converted = traits_type::from_bits(bits);
                packed_data.set_unsafe(i, mask_bits(traits_type::to_bits(converted), active_bpv));
            }
            size_ = view.count;
        }

        void clear() { size_ = 0; }

        static value_type max_value() { return packed_value_traits<value_type>::from_bits(COMPILED_MAX_BITS); }
        static constexpr uint8_t bits_per_element() { return BitsPerElement; }
        static constexpr size_t max_bits_value() { return COMPILED_MAX_BITS; }

        uint8_t get_bits_per_value() const { return packed_data.get_bpv(); }

        void set_bits_per_value(uint8_t bpv) {
            if (bpv == 0 || bpv > PackedArray<BitsPerElement>::WORD_BITS) {
                return;
            }
            if (bpv == packed_data.get_bpv()) {
                return;
            }
            init(bpv);
        }

        void shrink_to_fit() {
            if (size_ < capacity_) {
                size_type target = (size_ == 0) ? 1 : size_;
                const uint8_t active_bpv = packed_data.get_bpv();
                PackedArray<BitsPerElement> new_data(calc_words_for_bpv(target, active_bpv));
                new_data.set_bpv(active_bpv);
                new_data.copy_elements(packed_data, size_);
                packed_data = std::move(new_data);
                capacity_ = target;
            }
        }

        size_t memory_usage() const {
            const size_t words = calc_words_for_bpv(capacity_, packed_data.get_bpv());
            return words * sizeof(word_t);
        }

        bool operator==(const packed_vector& other) const {
            if (size_ != other.size_) {
                return false;
            }
            for (size_type i = 0; i < size_; ++i) {
                if (packed_data.get_unsafe(i) != other.packed_data.get_unsafe(i)) {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const packed_vector& other) const { return !(*this == other); }

        class iterator {
        private:
            packed_vector* parent = nullptr;
            size_type index = 0;

        public:
            friend class const_iterator;

            iterator() = default;

            iterator(packed_vector* p, size_type idx)
                : parent(p), index(idx) {}

            value_type operator*() const { return parent->from_storage_bits(parent->packed_data.get_unsafe(index)); }

            iterator& operator++() { ++index; return *this; }
            iterator operator++(int) { iterator tmp = *this; ++index; return tmp; }
            iterator& operator--() { --index; return *this; }
            iterator operator--(int) { iterator tmp = *this; --index; return tmp; }

            iterator operator+(size_type n) const { return iterator(parent, index + n); }
            iterator operator-(size_type n) const { return iterator(parent, index - n); }
            iterator& operator+=(size_type n) { index += n; return *this; }
            iterator& operator-=(size_type n) { index -= n; return *this; }

            bool operator==(const iterator& other) const { return index == other.index && parent == other.parent; }
            bool operator!=(const iterator& other) const { return !(*this == other); }
            bool operator<(const iterator& other) const { return index < other.index; }
            bool operator>(const iterator& other) const { return index > other.index; }
            bool operator<=(const iterator& other) const { return index <= other.index; }
            bool operator>=(const iterator& other) const { return index >= other.index; }

            std::ptrdiff_t operator-(const iterator& other) const {
                return static_cast<std::ptrdiff_t>(index) - static_cast<std::ptrdiff_t>(other.index);
            }

            size_type get_index() const { return index; }
        };

        class const_iterator {
        private:
            const packed_vector* parent = nullptr;
            size_type index = 0;

        public:
            const_iterator() = default;

            const_iterator(const packed_vector* p, size_type idx)
                : parent(p), index(idx) {}

            const_iterator(const iterator& it)
                : parent(it.parent), index(it.index) {}

            value_type operator*() const { return parent->from_storage_bits(parent->packed_data.get_unsafe(index)); }

            const_iterator& operator++() { ++index; return *this; }
            const_iterator operator++(int) { const_iterator tmp = *this; ++index; return tmp; }
            const_iterator& operator--() { --index; return *this; }
            const_iterator operator--(int) { const_iterator tmp = *this; --index; return tmp; }

            const_iterator operator+(size_type n) const { return const_iterator(parent, index + n); }
            const_iterator operator-(size_type n) const { return const_iterator(parent, index - n); }
            const_iterator& operator+=(size_type n) { index += n; return *this; }
            const_iterator& operator-=(size_type n) { index -= n; return *this; }

            bool operator==(const const_iterator& other) const { return index == other.index && parent == other.parent; }
            bool operator!=(const const_iterator& other) const { return !(*this == other); }
            bool operator<(const const_iterator& other) const { return index < other.index; }
            bool operator>(const const_iterator& other) const { return index > other.index; }
            bool operator<=(const const_iterator& other) const { return index <= other.index; }
            bool operator>=(const const_iterator& other) const { return index >= other.index; }

            std::ptrdiff_t operator-(const const_iterator& other) const {
                return static_cast<std::ptrdiff_t>(index) - static_cast<std::ptrdiff_t>(other.index);
            }

            size_type get_index() const { return index; }
        };

        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, size_); }
        const_iterator begin() const { return const_iterator(this, 0); }
        const_iterator end() const { return const_iterator(this, size_); }
        const_iterator cbegin() const { return const_iterator(this, 0); }
        const_iterator cend() const { return const_iterator(this, size_); }

        const word_t* data() const { return packed_data.raw_data(); }
        word_t* data() { return packed_data.raw_data(); }
            
        // Helper methods for optimized range copying and direct storage access
        const word_t* get_packed_data_words() const { return packed_data.raw_data(); }
        size_t words() const { return packed_data.words(); }
        word_t* raw_data() { return packed_data.raw_data(); }
        const word_t* raw_data() const { return packed_data.raw_data(); }
    };
    
    /*  
    ------------------------------------------------------------------------------------------------------------------
    --------------------------------------------------- ID_VECTOR ----------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------
    */

    template <typename T,  uint8_t BitsPerValue = 1>
    class ID_vector{
        static_assert(BitsPerValue > 0 && BitsPerValue <= 32, "BitsPerValue must be between 1 and 32");
    public:
        using count_type = uint32_t; // type for storing count of each ID
        using word_t = typename PackedArray<BitsPerValue>::word_t;
        
        // Index type mapping based on T
        using index_type = typename conditional_t<
            is_same_t<T, uint8_t>::value, uint8_t,
            typename conditional_t<
                is_same_t<T, uint16_t>::value, uint16_t,
                typename conditional_t<
                    is_same_t<T, uint32_t>::value, size_t,
                    typename conditional_t<
                        is_same_t<T, size_t>::value, size_t,
                        size_t  // Default to size_t if T is not recognized
                    >::type
                >::type
            >::type
        >::type;
        
        // Size type that can handle total count considering BitsPerValue
        // When BitsPerValue > 1, total size can exceed index_type capacity
        using size_type = typename conditional_t<
            (sizeof(index_type) <= 1), uint32_t,
            typename conditional_t<
                (sizeof(index_type) == 2), uint64_t,
                size_t
            >::type
        >::type;
        
    private:
        PackedArray<BitsPerValue> id_array; // BitsPerValue bits per ID
        index_type max_id_ = 0; // maximum ID that can be stored
        index_type min_id_ = 0; // minimum ID that can be stored
        size_type size_ = 0; // total number of ID instances stored

        // MAX_RF_ID based on index_type capacity
        constexpr static index_type MAX_RF_ID = 
            is_same_t<index_type, uint8_t>::value ? 255 :
            is_same_t<index_type, uint16_t>::value ? 65535 :
            2147483647; // max for size_t (assuming 32-bit signed)
            
        constexpr static index_type DEFAULT_MAX_ID = 
            is_same_t<index_type, uint8_t>::value ? 63 :
            is_same_t<index_type, uint16_t>::value ? 255 :
            127; // default max ID
            
    constexpr static count_type MAX_COUNT = (static_cast<count_type>(1ULL) << BitsPerValue) - 1; // maximum count per ID

    static constexpr size_t bits_to_words(size_t bits){ const size_t word_bits = PackedArray<BitsPerValue>::WORD_BITS; return (bits + word_bits - 1) / word_bits; }

        void allocate_bits(){
            const size_t range = static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1; // number of IDs in range
            const size_t total_bits = range * BitsPerValue; // multiply by bits per value
            const size_t words = bits_to_words(total_bits);
            id_array = PackedArray<BitsPerValue>(words);
        }

        // Convert external ID to internal array index
        index_type inline id_to_index(index_type id) const {
            return id - min_id_;
        }

        // Convert internal array index to external ID
        index_type inline index_to_id(index_type index) const {
            return index + min_id_;
        }

        index_type first_active_id() const {
            if (size_ == 0) {
                throw std::out_of_range("ID_vector is empty");
            }

            if constexpr (BitsPerValue == 1) {
                const word_t* words = id_array.raw_data();
                const size_t word_count = id_array.words();
                const size_t first_word = detail_simd::first_nonzero_word(words, word_count);
                if (first_word != std::numeric_limits<size_t>::max()) {
                    const size_t bit_index = first_word * PackedArray<BitsPerValue>::WORD_BITS + detail_simd::trailing_zeros(words[first_word]);
                    const size_t range = static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1;
                    if (bit_index < range) {
                        return static_cast<index_type>(static_cast<size_t>(min_id_) + bit_index);
                    }
                }
            }

            for (index_type id = min_id_; id <= max_id_; ++id) {
                if (id_array.get(id_to_index(id)) > 0) {
                    return id;
                }
            }
            throw std::out_of_range("ID_vector::first_active_id() internal error");
        }

        index_type last_active_id() const {
            if (size_ == 0) {
                throw std::out_of_range("ID_vector is empty");
            }

            if constexpr (BitsPerValue == 1) {
                const word_t* words = id_array.raw_data();
                const size_t word_count = id_array.words();
                const size_t last_word = detail_simd::last_nonzero_word(words, word_count);
                if (last_word != std::numeric_limits<size_t>::max()) {
                    const size_t value = words[last_word];
                    const unsigned msb = static_cast<unsigned>(std::numeric_limits<size_t>::digits - 1u - detail_simd::leading_zeros(value));
                    const size_t bit_index = last_word * PackedArray<BitsPerValue>::WORD_BITS + msb;
                    const size_t range = static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1;
                    if (bit_index < range) {
                        return static_cast<index_type>(static_cast<size_t>(min_id_) + bit_index);
                    }
                }
            }

            for (index_type id = max_id_;; --id) {
                if (id_array.get(id_to_index(id)) > 0) {
                    return id;
                }
                if (id == min_id_) {
                    break;
                }
            }
            throw std::out_of_range("ID_vector::last_active_id() internal error");
        }

    public:
        // Set maximum ID that can be stored and allocate memory accordingly
        void set_maxID(index_type new_max_id) {
            if(new_max_id > MAX_RF_ID){
                throw std::out_of_range("Max RF ID exceeds limit");
            }
            if(new_max_id < min_id_){
                throw std::out_of_range("Max ID cannot be less than min ID");
            }
            
            // If vector is empty, just update max_id and allocate new memory
            if(size_ == 0) {
                max_id_ = new_max_id;
                allocate_bits();
                return;
            }
            
            // Vector has elements - check if we can safely preserve data
            index_type current_max_element = maxID(); // Get largest actual element
            
            if(new_max_id >= current_max_element) {
                // Safe case: new max_id is at or above the largest element
                // We can preserve all data by copying to new memory layout
                
                // Save current data
                index_type old_max_id = max_id_;
                const size_t old_words = id_array.words();
                PackedArray<BitsPerValue> old_array(old_words);
                old_array.copy_from(id_array, old_words);
                
                // Update max_id and allocate new memory
                max_id_ = new_max_id;
                allocate_bits();
                
                // Copy elements from old array to new array (indices remain the same)
                for(index_type old_id = min_id_; old_id <= old_max_id; ++old_id) {
                    index_type old_index = old_id - min_id_;
                    count_type element_count = old_array.get(old_index);
                    if(element_count > 0) {
                        index_type new_index = old_id - min_id_; // Same index in new array
                        id_array.set(new_index, element_count);
                    }
                }
                // size_ remains the same since we preserved all elements
                
            } else {
                // Potentially unsafe case: new max_id is below some existing elements
                // This would cause data loss, so we throw an exception
                throw std::out_of_range("Cannot set max_id below existing elements. Current largest element is " + std::to_string(current_max_element));
            }
        }
        
        // Set minimum ID that can be stored and allocate memory accordingly
        void set_minID(index_type new_min_id) {
            if(new_min_id > MAX_RF_ID){
                throw std::out_of_range("Min RF ID exceeds limit");
            }
            if(new_min_id > max_id_){
                throw std::out_of_range("Min ID cannot be greater than max ID");
            }
            
            // If vector is empty, just update min_id and allocate new memory
            if(size_ == 0) {
                min_id_ = new_min_id;
                allocate_bits();
                return;
            }
            
            // Vector has elements - check if we can safely cut off lower range
            index_type current_min_element = minID(); // Get smallest actual element
            
            if(new_min_id <= current_min_element) {
                // Safe case: new min_id is at or below the smallest element
                // We can preserve all data by copying to new memory layout
                
                // Save current data
                index_type old_min_id = min_id_;
                const size_t old_words = id_array.words();
                PackedArray<BitsPerValue> old_array(old_words);
                old_array.copy_from(id_array, old_words);
                
                // Update min_id and allocate new memory
                min_id_ = new_min_id;
                allocate_bits();
                
                // Copy elements from old array to new array with adjusted indices
                for(index_type old_id = current_min_element; old_id <= max_id_; ++old_id) {
                    index_type old_index = old_id - old_min_id;
                    count_type element_count = old_array.get(old_index);
                    if(element_count > 0) {
                        index_type new_index = old_id - min_id_;
                        id_array.set(new_index, element_count);
                    }
                }
                // size_ remains the same since we preserved all elements
                
            } else {
                // Potentially unsafe case: new min_id is above some existing elements
                // This would cause data loss, so we throw an exception
                throw std::out_of_range("Cannot set min_id above existing elements. Current smallest element is " + std::to_string(current_min_element));
            }
        }

        // Set both min and max ID range and allocate memory accordingly
        void set_ID_range(index_type new_min_id, index_type new_max_id) {
            if(new_min_id > MAX_RF_ID || new_max_id > MAX_RF_ID){
                throw std::out_of_range("RF ID exceeds limit");
            }
            if(new_min_id > new_max_id){
                throw std::out_of_range("Min ID cannot be greater than max ID");
            }
            
            // If vector is empty, just update range and allocate new memory
            if(size_ == 0) {
                min_id_ = new_min_id;
                max_id_ = new_max_id;
                allocate_bits();
                return;
            }
            
            // Vector has elements - check if we can safely preserve data
            index_type current_min_element = minID(); // Get smallest actual element
            index_type current_max_element = maxID(); // Get largest actual element
            
            if(new_min_id <= current_min_element && new_max_id >= current_max_element) {
                // Safe case: new range encompasses all existing elements
                // We can preserve all data by copying to new memory layout
                
                // Save current data
                index_type old_min_id = min_id_;
                index_type old_max_id = max_id_;
                const size_t old_words = id_array.words();
                PackedArray<BitsPerValue> old_array(old_words);
                old_array.copy_from(id_array, old_words);
                
                // Update range and allocate new memory
                min_id_ = new_min_id;
                max_id_ = new_max_id;
                allocate_bits();
                
                // Copy elements from old array to new array with adjusted indices
                for(index_type old_id = old_min_id; old_id <= old_max_id; ++old_id) {
                    index_type old_index = old_id - old_min_id;
                    count_type element_count = old_array.get(old_index);
                    if(element_count > 0) {
                        index_type new_index = old_id - min_id_;
                        id_array.set(new_index, element_count);
                    }
                }
                // size_ remains the same since we preserved all elements
                
            } else {
                // Potentially unsafe case: new range doesn't encompass all existing elements
                char error_msg[128];
                snprintf(error_msg, sizeof(error_msg), "Cannot set ID range that excludes existing elements. Current elements range: [%llu, %llu]", static_cast<unsigned long long>(current_min_element), static_cast<unsigned long long>(current_max_element));
                throw std::out_of_range(error_msg);
            }
        }

        // default constructor (default max ID 127, min ID 0 -> 128 bits -> 16 bytes)
        ID_vector(){
            set_maxID(DEFAULT_MAX_ID);
        }

        // constructor with max expected ID - calls set_maxID automatically
        explicit ID_vector(index_type max_id){
            set_maxID(max_id);
        }

        // constructor with min and max expected ID range
        ID_vector(index_type min_id, index_type max_id){
            set_ID_range(min_id, max_id);
        }

        // Copy constructor
        ID_vector(const ID_vector& other) 
            : id_array(), max_id_(other.max_id_), min_id_(other.min_id_), size_(other.size_) {
            const size_t words = other.id_array.words();
            id_array = PackedArray<BitsPerValue>(words);
            id_array.copy_from(other.id_array, words);
        }

        // constructor with vector of IDs (uint8_t, uint16_t, uint32_t, size_t)
        template<typename Y>
        ID_vector(const vector<Y>& ids,
                  typename std::enable_if<std::is_same<Y, uint8_t>::value || 
                                         std::is_same<Y, uint16_t>::value || 
                                         std::is_same<Y, uint32_t>::value ||
                                         std::is_same<Y, size_t>::value >::type* = nullptr) {
            if(ids.empty()){
                set_maxID(DEFAULT_MAX_ID);
                return;
            }
            ids.sort();
            index_type min_id = static_cast<size_t>(ids.front());
            index_type max_id = static_cast<size_t>(ids.back());
            set_ID_range(min_id, max_id);
            for(const Y& id : ids){
                push_back(static_cast<size_t>(id));
            }
        }

        // Move constructor
        ID_vector(ID_vector&& other) noexcept 
            : id_array(std::move(other.id_array)), 
              max_id_(other.max_id_), min_id_(other.min_id_), size_(other.size_) {
            other.min_id_ = 0;
            other.max_id_ = 0;
            other.size_ = 0;
        }

        // Copy assignment operator
        ID_vector& operator=(const ID_vector& other) {
            if (this != &other) {
                min_id_ = other.min_id_;
                max_id_ = other.max_id_;
                size_ = other.size_;
                
                const size_t words = other.id_array.words();
                id_array = PackedArray<BitsPerValue>(words);
                id_array.copy_from(other.id_array, words);
            }
            return *this;
        }

        // Move assignment operator
        ID_vector& operator=(ID_vector&& other) noexcept {
            if (this != &other) {
                id_array = std::move(other.id_array);
                min_id_ = other.min_id_;
                max_id_ = other.max_id_;
                size_ = other.size_;
                
                other.min_id_ = 0;
                other.max_id_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        // Assignment from another ID_vector with different BitsPerValue
        template<uint8_t OtherBits>
        ID_vector& operator=(const ID_vector<T, OtherBits>& other) {
            clear();
            if (other.empty()) return *this;
            
            min_id_ = other.minID();
            max_id_ = other.maxID();
            allocate_bits();
            
            for (index_type id = min_id_; id <= max_id_; ++id) {
                count_type c = other.count(id);
                if (c > 0) {
                    // Cap count to MAX_COUNT of this ID_vector
                    count_type capped_c = (c > MAX_COUNT) ? MAX_COUNT : c;
                    id_array.set(id_to_index(id), capped_c);
                    size_ += capped_c;
                }
            }
            return *this;
        }

        // Destructor (default is fine since PackedArray handles its own cleanup)
        ~ID_vector() = default;


        // check presence
        bool contains(index_type id) const {
            if(id < min_id_ || id > max_id_) return false;
            return id_array.get(id_to_index(id)) != 0;
        }

        // insert ID (order independent, data structure is inherently sorted)
        void push_back(index_type id){
            // Check if ID exceeds absolute maximum
            if(id > MAX_RF_ID){
                throw std::out_of_range("ID exceeds maximum allowed RF ID limit");
            }
            
            // Auto-expand range if necessary
            if(id > max_id_){
                set_maxID(id);
            } else if(id < min_id_){
                set_minID(id);
            }
            
            index_type index = id_to_index(id);
            count_type current_count = id_array.get(index);
            if(current_count < MAX_COUNT){
                id_array.set(index, current_count + 1);
                ++size_;
            } // if already at max count, ignore (do nothing)
        }

        // get count of specific ID
        count_type count(index_type id) const {
            if(id < min_id_ || id > max_id_) return 0;
            return id_array.get(id_to_index(id));
        }

        // remove one instance of specific ID (if exists)
        bool erase(index_type id){
            if(id < min_id_ || id > max_id_) return false;
            index_type index = id_to_index(id);
            count_type current_count = id_array.get(index);
            if(current_count > 0){
                id_array.set(index, current_count - 1);
                --size_;
                return true;
            }
            return false;
        }

        // largest ID in the vector (if empty, throws)
        index_type back() const {
            return last_active_id();
        }

        // pop largest ID (remove one instance)
        void pop_back(){
            if(size_ == 0) return; // empty

            const index_type id = last_active_id();
            const index_type index = id_to_index(id);
            const count_type current_count = id_array.get(index);
            id_array.set(index, current_count - 1);
            --size_;
        }

        T front() const {
            return first_active_id();
        }

        void pop_front() {
            if(size_ == 0) return; // empty

            const index_type id = first_active_id();
            const index_type index = id_to_index(id);
            const count_type current_count = id_array.get(index);
            id_array.set(index, current_count - 1);
            --size_;
        }

        void reserve(index_type new_max_id){
            if(new_max_id >= MAX_RF_ID){
                throw std::out_of_range("Max RF ID exceeds limit");
            }
            if(new_max_id < min_id_){
                throw std::out_of_range("Max ID cannot be less than min ID");
            }
            if(new_max_id > max_id_){
                set_maxID(new_max_id);
            }
        }

        // get number of unique IDs stored (if bitspervalue=1, this is same as size())
        size_type unique_size() const {
            if(BitsPerValue == 1) return size_;
            size_type unique_count = 0;
            index_type range = (size_t)max_id_  -  (size_t)min_id_ + 1;
            for(index_type i = 0; i < range; ++i){
                if(id_array.get(i) > 0) ++unique_count;
            }
            return unique_count;
        }

        // nth element (0-based) among all ID instances (in ascending order)
        // When an ID appears multiple times, it will be returned multiple times
        index_type operator[](size_type index) const {
            if(index >= size_) throw std::out_of_range("ID_vector::operator[] index out of range");
            
            size_type current_count = 0;
            for(index_type id = min_id_; id <= max_id_; ++id) {
                count_type id_count = id_array.get(id_to_index(id));
                if(id_count > 0) {
                    if(current_count + id_count > index) {
                        // The index falls within this ID's instances
                        return id;
                    }
                    current_count += id_count;
                }
            }
            throw std::out_of_range("ID_vector::operator[] internal error");
        }

        // iterator over all ID instances (ascending order with repetitions)
        class iterator {
            const ID_vector* vec = nullptr;
            index_type current_id = 0; // Current ID being processed
            count_type remaining_count = 0; // Remaining instances of current ID

            void find_first() {
                current_id = vec ? vec->min_id_ : 0;
                remaining_count = 0;
                
                if (!vec) {
                    current_id = 0;
                    remaining_count = 0;
                    return;
                }

                // Find the first ID with count > 0, starting from min_id
                while (current_id <= vec->max_id_) {
                    count_type id_count = vec->id_array.get(vec->id_to_index(current_id));
                    if (id_count > 0) {
                        remaining_count = id_count - 1; // -1 because we're returning this instance
                        return;
                    }
                    ++current_id;
                }

                // No IDs found
                current_id = vec->max_id_ + 1;
                remaining_count = 0;
            }

            void find_next() {
                if (!vec) {
                    current_id = 0;
                    remaining_count = 0;
                    return;
                }

                // If we still have instances of the current ID, just decrement
                if (remaining_count > 0) {
                    --remaining_count;
                    return;
                }

                // Find the next ID with count > 0
                ++current_id; // Move to next ID
                while (current_id <= vec->max_id_) {
                    count_type id_count = vec->id_array.get(vec->id_to_index(current_id));
                    if (id_count > 0) {
                        remaining_count = id_count - 1; // -1 because we're returning this instance
                        return;
                    }
                    ++current_id;
                }

                // No more IDs found
                current_id = vec->max_id_ + 1;
                remaining_count = 0;
            }

        public:
            using value_type = index_type;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;
            using pointer = const value_type*;
            using reference = const value_type&;

            iterator() : vec(nullptr), current_id(0), remaining_count(0) {}

            // Constructor for begin() and end()
            iterator(const ID_vector* v, bool is_end) : vec(v), current_id(0), remaining_count(0) {
                if (!v || v->size_ == 0 || is_end) {
                    current_id = v ? v->max_id_ + 1 : 0;
                    remaining_count = 0;
                } else {
                    find_first();
                }
            }

            reference operator*() const { return current_id; }

            iterator& operator++() {
                find_next();
                return *this;
            }

            iterator operator++(int) {
                iterator tmp = *this;
                find_next();
                return tmp;
            }

            bool operator==(const iterator& other) const {
                return vec == other.vec && current_id == other.current_id && remaining_count == other.remaining_count;
            }

            bool operator!=(const iterator& other) const {
                return !(*this == other);
            }
        };

        iterator begin() const { return iterator(this, false); }
        iterator end() const { return iterator(this, true); }

        // Comparison operators
        bool operator==(const ID_vector& other) const {
            if (this == &other) return true;
            if (min_id_ != other.min_id_ || max_id_ != other.max_id_ || size_ != other.size_) {
                return false;
            }
            
            // Compare element by element
            for (index_type id = min_id_; id <= max_id_; ++id) {
                if (count(id) != other.count(id)) {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const ID_vector& other) const {
            return !(*this == other);
        }

        // Subset comparison (this ⊆ other)
        bool is_subset_of(const ID_vector& other) const {
            if (min_id_ < other.min_id_ || max_id_ > other.max_id_) {
                return false; // Range not contained
            }
            
            for (index_type id = min_id_; id <= max_id_; ++id) {
                if (count(id) > other.count(id)) {
                    return false; // This has more instances than other
                }
            }
            return true;
        }

        // Set operations
        ID_vector operator|(const ID_vector& other) const { // Union
            index_type new_min = (min_id_ < other.min_id_) ? min_id_ : other.min_id_;
            index_type new_max = (max_id_ > other.max_id_) ? max_id_ : other.max_id_;
            ID_vector result(new_min, new_max);
            
            for (index_type id = new_min; id <= new_max; ++id) {
                count_type count1 = (id >= min_id_ && id <= max_id_) ? count(id) : 0;
                count_type count2 = (id >= other.min_id_ && id <= other.max_id_) ? other.count(id) : 0;
                count_type max_count = (count1 > count2) ? count1 : count2;
                
                for (count_type i = 0; i < max_count; ++i) {
                    result.push_back(id);
                }
            }
            return result;
        }

        ID_vector operator&(const ID_vector& other) const { // Intersection
            index_type new_min = (min_id_ > other.min_id_) ? min_id_ : other.min_id_;
            index_type new_max = (max_id_ < other.max_id_) ? max_id_ : other.max_id_;
            
            if (new_min > new_max) {
                return ID_vector(); // Empty intersection
            }
            
            ID_vector result(new_min, new_max);
            
            for (index_type id = new_min; id <= new_max; ++id) {
                count_type count1 = count(id);
                count_type count2 = other.count(id);
                count_type min_count = (count1 < count2) ? count1 : count2;
                
                for (count_type i = 0; i < min_count; ++i) {
                    result.push_back(id);
                }
            }
            return result;
        }

        // Compound assignment operators
        ID_vector& operator|=(const ID_vector& other) { // Union assignment
            *this = *this | other;
            return *this;
        }

        ID_vector& operator&=(const ID_vector& other) { // Intersection assignment
            *this = *this & other;
            return *this;
        }

        // Fill vector with all values in the current range [min_id_, max_id_]
        // For BitsPerValue > 1, fills with maximum count (MAX_COUNT) for each ID
        void fill() {
            if (max_id_ < min_id_) return; // Invalid range
            
            clear(); // Start fresh
            
            for (index_type id = min_id_; id <= max_id_; ++id) {
                // Fill with maximum possible count for each ID
                for (count_type i = 0; i < MAX_COUNT; ++i) {
                    push_back(id);
                }
            }
        }
        // remove all instances of specific ID (if exists)
        bool erase_all(index_type id){
            if(id < min_id_ || id > max_id_) return false;
            index_type index = id_to_index(id);
            count_type current_count = id_array.get(index);
            if(current_count > 0){
                id_array.set(index, 0);
                size_ -= current_count; // subtract all instances
                return true;
            }
            return false;
        }

        // Erase all instances of IDs in range [start, end] (inclusive)
        // Does NOT change the vector's min_id_/max_id_ range
        void erase_range(index_type start, index_type end) {
            if (start > end) return; // Invalid range
            
            // Only process IDs within our current range
            index_type actual_start = (start > min_id_) ? start : min_id_;
            index_type actual_end = (end < max_id_) ? end : max_id_;
            
            if (actual_start > actual_end) return; // No overlap
            
            for (index_type id = actual_start; id <= actual_end; ++id) {
                erase_all(id); // Remove all instances of this ID
            }
        }

        // Insert all IDs in range [start, end] (inclusive), one instance each
        // DOES allow expansion of the vector's min_id_/max_id_ range
        void insert_range(index_type start, index_type end) {
            if (start > end) return; // Invalid range
            
            for (index_type id = start; id <= end; ++id) {
                push_back(id); // This will auto-expand range if needed
            }
        }

        // Addition operator: adds one instance of each ID from other vector
        // Static assertion ensures compatible BitsPerValue
        template<uint8_t OtherBitsPerValue>
        ID_vector operator+(const ID_vector<T, OtherBitsPerValue>& other) const {
            static_assert(BitsPerValue == OtherBitsPerValue, 
                         "Cannot perform arithmetic operations on ID_vectors with different BitsPerValue");
            
            // Create result with expanded range to accommodate both vectors
            index_type new_min = (min_id_ < other.get_minID()) ? min_id_ : other.get_minID();
            index_type new_max = (max_id_ > other.get_maxID()) ? max_id_ : other.get_maxID();
            
            // Handle empty vectors
            if (size_ == 0 && other.size() == 0) {
                return ID_vector();
            } else if (size_ == 0) {
                new_min = other.get_minID();
                new_max = other.get_maxID();
            } else if (other.size() == 0) {
                new_min = min_id_;
                new_max = max_id_;
            }
            
            ID_vector result(new_min, new_max);
            
            // Copy this vector's elements
            for (index_type id = min_id_; id <= max_id_; ++id) {
                count_type my_count = count(id);
                for (count_type i = 0; i < my_count; ++i) {
                    result.push_back(id);
                }
            }
            
            // Add one instance of each ID from other vector
            for (index_type id = other.get_minID(); id <= other.get_maxID(); ++id) {
                if (other.count(id) > 0) {
                    result.push_back(id); // Add one instance
                }
            }
            
            return result;
        }

        // Subtraction operator: removes all instances of IDs present in other vector
        // Static assertion ensures compatible BitsPerValue
        template<uint8_t OtherBitsPerValue>
        ID_vector operator-(const ID_vector<T, OtherBitsPerValue>& other) const {
            static_assert(BitsPerValue == OtherBitsPerValue, 
                         "Cannot perform arithmetic operations on ID_vectors with different BitsPerValue");
            
            ID_vector result(min_id_, max_id_);
            
            for (index_type id = min_id_; id <= max_id_; ++id) {
                count_type my_count = count(id);
                if (my_count > 0) {
                    // If other vector contains this ID, remove all instances
                    bool other_has_id = (id >= other.get_minID() && id <= other.get_maxID() && other.count(id) > 0);
                    if (!other_has_id) {
                        // Keep all instances if other doesn't have this ID
                        for (count_type i = 0; i < my_count; ++i) {
                            result.push_back(id);
                        }
                    }
                    // If other has this ID, don't add any instances (remove all)
                }
            }
            
            return result;
        }

        // Addition assignment operator: adds one instance of each ID from other vector
        template<uint8_t OtherBitsPerValue>
        ID_vector& operator+=(const ID_vector<T, OtherBitsPerValue>& other) {
            static_assert(BitsPerValue == OtherBitsPerValue, 
                         "Cannot perform arithmetic operations on ID_vectors with different BitsPerValue");
            
            // Add one instance of each ID from other vector
            for (index_type id = other.get_minID(); id <= other.get_maxID(); ++id) {
                if (other.count(id) > 0) {
                    push_back(id); // Add one instance (auto-expands range if needed)
                }
            }
            return *this;
        }

        // Subtraction assignment operator: removes all instances of IDs present in other vector
        template<uint8_t OtherBitsPerValue>
        ID_vector& operator-=(const ID_vector<T, OtherBitsPerValue>& other) {
            static_assert(BitsPerValue == OtherBitsPerValue, 
                         "Cannot perform arithmetic operations on ID_vectors with different BitsPerValue");
            
            // Remove all instances of IDs present in other vector
            for (index_type id = other.get_minID(); id <= other.get_maxID(); ++id) {
                if (other.count(id) > 0) {
                    erase_all(id); // Remove all instances of this ID
                }
            }
            return *this;
        }

       // number of stored IDs
        size_type size() const { return size_; }
        bool empty() const { return size_ == 0; }

        void clear(){
            if(size_ == 0) return; // Already empty
            
            auto* data = id_array.raw_data();
            if(data != nullptr) {
                const size_t words = id_array.words();
                detail_simd::fill_words(data, words, static_cast<size_t>(0));
            }
            size_ = 0;
        }
        void shrink_to_fit() {    
            // Fit the ID_vector to the current range and size
            if(size_ == 0) {
                // Empty vector - nothing to fit
                return;
            }
            index_type new_min_id = minID();
            index_type new_max_id = maxID();
            if(new_min_id != min_id_ || new_max_id != max_id_) {
                set_ID_range(new_min_id, new_max_id);
            }
        }


        // Get current minimum ID that can be stored
        index_type get_minID() const {
            return min_id_;
        }

        // Get current maximum ID that can be stored  
        index_type get_maxID() const {
            return max_id_;
        }

        // get the smallest ID currently stored in the vector
        T minID() const {
            return first_active_id();
        }

        // get the largest ID currently stored in the vector
        index_type maxID() const {
            return last_active_id();
        }

        size_t capacity() const {
            index_type range = (size_t)max_id_  -  (size_t)min_id_ + 1;
            return range;
        }

        size_t memory_usage() const {
            const size_t range = static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1;
            const size_t total_bits = range * BitsPerValue;
            const size_t words = bits_to_words(total_bits);
            const size_t bytes = words * sizeof(word_t);
            return sizeof(ID_vector) + bytes;
        }

        /**
         * @brief Get the current bits per value (runtime value).
         * @return Current bits per value setting.
         */
        uint8_t get_bits_per_value() const noexcept {
            return id_array.get_bpv();
        }

        /**
         * @brief Set the bits per value dynamically.
         * @param new_bpv New bits per value (must be 1-32).
         * @return true if successful, false if new_bpv is invalid or would cause data loss.
         * @note This reallocates the internal array and preserves existing data if possible.
         *       Will fail if any existing count value exceeds the new maximum count.
         */
        bool set_bits_per_value(uint8_t new_bpv) noexcept {
            // Validate new bits per value
            if (new_bpv == 0 || new_bpv > 32) {
                return false;
            }

            // Get current bits per value
            uint8_t current_bpv = id_array.get_bpv();
            
            // If same, no change needed
            if (new_bpv == current_bpv) {
                return true;
            }

            // Calculate new max count
            count_type new_max_count = (new_bpv >= 32) 
                ? std::numeric_limits<count_type>::max() 
                : ((static_cast<count_type>(1ULL) << new_bpv) - 1);

            // If reducing bits, check if any existing values would overflow
            if (new_bpv < current_bpv && size_ > 0) {
                index_type range = static_cast<index_type>((static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1));
                for (index_type i = 0; i < range; ++i) {
                    count_type current_count = id_array.get(i);
                    if (current_count > new_max_count) {
                        // Would cause data loss
                        return false;
                    }
                }
            }

            // Save old data
            const size_t old_words = id_array.words();
            PackedArray<BitsPerValue> old_array(old_words);
            old_array.copy_from(id_array, old_words);

            // Calculate new word count and reallocate
            const size_t range = static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1;
            const size_t new_total_bits = range * new_bpv;
            const size_t new_words = bits_to_words(new_total_bits);
            
            id_array = PackedArray<BitsPerValue>(new_words);
            id_array.set_bpv(new_bpv);

            // Copy elements from old array to new array with new bit width
            if (size_ > 0) {
                index_type element_range = static_cast<index_type>((static_cast<size_t>(max_id_) - static_cast<size_t>(min_id_) + 1));
                for (index_type i = 0; i < element_range; ++i) {
                    count_type count_val = old_array.get(i);
                    if (count_val > 0) {
                        id_array.set(i, count_val);
                    }
                }
            }

            return true;
        }
        
    };

}   // namespace MCU