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
 * @tparam max_size
 */
template <typename T, u64 max_size> struct StaticVector
{
    using value_type     = T;
    using size_type      = u64;
    using iterator       = typename std::array<T, max_size>::iterator;
    using const_iterator = typename std::array<T, max_size>::const_iterator;

    constexpr explicit StaticVector(u8 n = 0) : _size{n}
    {
        if (_size > max_size) { throw std::overflow_error("SmallVector overflow"); }
    }

    constexpr StaticVector(const StaticVector& other)     = default;
    constexpr StaticVector(StaticVector&& other) noexcept = default;

    constexpr explicit StaticVector(std::initializer_list<T> init) : _size{init.size()}
    {
        for (u8 i = 0; i < _size; i++) { _storage[i] = init[i]; }
    }

    constexpr void push_back(T val)
    {
        _storage[_size++] = val;
        if (_size > max_size) { throw std::overflow_error("SmallVector overflow"); }
    }

    constexpr void pop_back()
    {
        if (_size == 0) { throw std::underflow_error("SmallVector underflow"); }
        back().~T(); // call destructor
        _size--;
    }

    constexpr u64 size() const noexcept { return _size; }
    constexpr u64 capacity() const noexcept { return max_size; }

    constexpr void clear()
    {
        while (_size > 0) { pop_back(); }
    }

    constexpr T& front() noexcept { return _storage.front(); }
    constexpr const T& front() const noexcept { return _storage.front(); }

    constexpr T& back() noexcept { return _storage[_size - 1]; }
    constexpr const T& back() const noexcept { return _storage[_size - 1]; }

    constexpr iterator begin() noexcept { return _storage.begin(); }
    constexpr const_iterator begin() const noexcept { return _storage.begin(); }

    constexpr iterator end() noexcept { return _storage.end(); }
    constexpr const_iterator end() const noexcept { return _storage.end(); }

    constexpr T& operator[](u8 index) noexcept { return _storage[index]; }
    constexpr const T& operator[](u8 index) const noexcept { return _storage[index]; }

    constexpr T& data() noexcept { return _storage.data(); }
    constexpr const T& data() const noexcept { return _storage.data(); }

  private:
    std::array<T, max_size> _storage;
    u8 _size = 0;
};
} // namespace sm
