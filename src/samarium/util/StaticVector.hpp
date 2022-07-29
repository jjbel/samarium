/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>            // for array
#include <initializer_list> // for initializer_list
#include <stdexcept>        // for overflow_error, underflow_error

#include "samarium/core/types.hpp"

namespace sm
{
/**
 * @brief               Stack-based, Fixed Capacity, Contiguous Resizable Container. From
 * https://cpp-optimizations.netlify.app/small_vectors/
 *
 * @tparam T
 * @tparam capacity
 */
template <typename T, u64 capacity> class StaticVector
{
  public:
    using value_type     = T;
    using size_type      = u64;
    using iterator       = typename std::array<T, capacity>::iterator;
    using const_iterator = typename std::array<T, capacity>::const_iterator;

    explicit StaticVector(u8 n = 0) : _size{n}
    {
        if (_size > capacity) { throw std::overflow_error("SmallVector overflow"); }
    }

    StaticVector(const StaticVector& other)     = default;
    StaticVector(StaticVector&& other) noexcept = default;

    explicit StaticVector(std::initializer_list<T> init) : _size{init.size()}
    {
        for (u8 i = 0; i < _size; i++) { _storage[i] = init[i]; }
    }

    void push_back(T val)
    {
        _storage[_size++] = val;
        if (_size > capacity) { throw std::overflow_error("SmallVector overflow"); }
    }

    void pop_back()
    {
        if (_size == 0) { throw std::underflow_error("SmallVector underflow"); }
        back().~T(); // call destructor
        _size--;
    }

    u64 size() const noexcept { return _size; }

    void clear()
    {
        while (_size > 0) { pop_back(); }
    }

    T& front() noexcept { return _storage.front(); }
    const T& front() const noexcept { return _storage.front(); }

    T& back() noexcept { return _storage[_size - 1]; }
    const T& back() const noexcept { return _storage[_size - 1]; }

    iterator begin() noexcept { return _storage.begin(); }
    const_iterator begin() const noexcept { return _storage.begin(); }

    iterator end() noexcept { return _storage.end(); }
    const_iterator end() const noexcept { return _storage.end(); }

    T& operator[](u8 index) noexcept { return _storage[index]; }
    const T& operator[](u8 index) const noexcept { return _storage[index]; }

    T& data() noexcept { return _storage.data(); }
    const T& data() const noexcept { return _storage.data(); }

  private:
    std::array<T, capacity> _storage;
    u8 _size = 0;
};
} // namespace sm
