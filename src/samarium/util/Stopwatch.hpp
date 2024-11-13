/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <chrono> // for duration

#include "samarium/core/types.hpp" // for f64

namespace sm
{
struct Stopwatch
{
    using Duration = std::chrono::duration<f64>;

    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};

    void reset();

    [[nodiscard]] auto time() const -> Duration;

    [[nodiscard]] auto seconds() const -> f64;

    [[nodiscard]] auto current_fps() -> f64;

    void print() const;

    std::string str_ms() const;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_STOPWATCH_IMPL)

#include "fmt/format.h"

#include "samarium/core/inline.hpp" // for SM_INLINE

namespace sm
{
SM_INLINE void Stopwatch::reset() { start = std::chrono::steady_clock::now(); }

SM_INLINE auto Stopwatch::time() const -> Stopwatch::Duration
{
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(finish - start);
}

[[nodiscard]] SM_INLINE auto Stopwatch::seconds() const -> f64 { return this->time().count(); }

[[nodiscard]] SM_INLINE auto Stopwatch::current_fps() -> f64
{
    const auto sec = seconds();
    reset();
    return 1.0 / sec;
}

SM_INLINE std::string Stopwatch::str_ms() const
{
    return fmt::format("{:.3f}ms", this->seconds() * 1000.0);
}

SM_INLINE void Stopwatch::print() const { fmt::print("{:.3}ms\n", this->seconds() * 1000.0); }
}; // namespace sm

#endif
