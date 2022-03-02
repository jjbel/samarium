/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <chrono>
#include <concepts>
#include <iterator>
#include <string>

#include "fmt/format.h"

#include "samarium/core/types.hpp"

namespace sm::util
{
[[nodiscard]] consteval size_t strlen(const char* str) // NOSONAR
{
    return *str ? 1u + strlen(str + 1) : 0u;
}

[[nodiscard]] consteval u8 hex_to_int_safe(char ch)
{
    if ('0' <= ch && ch <= '9') return static_cast<u8>(ch - '0');
    if ('a' <= ch && ch <= 'f') return static_cast<u8>(ch - 'a' + 10);
    if ('A' <= ch && ch <= 'F') return static_cast<u8>(ch - 'A' + 10);
    throw std::invalid_argument("hex character must be 0-9, a-f, or A-F");
}

std::string get_date_filename(const std::string& extension);

class Stopwatch
{
  public:
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};

    auto reset() { start = std::chrono::steady_clock::now(); }

    auto time() const
    {
        const auto finish = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<f64>>(finish -
                                                                      start);
    }

    auto print() const
    {
        fmt::print("Took {:.3}ms\n", this->time().count() * 1000);
    }
};
} // namespace sm::util
