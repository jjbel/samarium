/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <chrono> // for duration

#include "fmt/core.h" // for print

#include "samarium/core/types.hpp"

namespace sm
{
struct Stopwatch
{
    using Duration_t = std::chrono::duration<f64>;

    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};

    void reset();

    [[nodiscard]] auto time() const -> Duration_t;

    [[nodiscard]] auto seconds() const -> f64;

    [[nodiscard]] auto current_fps() -> f64;

    void print() const;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_STOPWATCH_IMPL)

#include "fmt/format.h"

#include "samarium/core/inline.hpp" // for SM_INLINE

namespace sm
{
SM_INLINE void Stopwatch::reset() { start = std::chrono::steady_clock::now(); }

SM_INLINE auto Stopwatch::time() const -> Stopwatch::Duration_t
{
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration_t>(finish - start);
}

[[nodiscard]] SM_INLINE auto Stopwatch::seconds() const -> f64 { return this->time().count(); }

[[nodiscard]] SM_INLINE auto Stopwatch::current_fps() -> f64
{
    const auto sec = seconds();
    reset();
    return 1.0 / sec;
}

SM_INLINE void Stopwatch::print() const { fmt::print("{:.3}ms\n", this->time().count() * 1000.0); }
}; // namespace sm

#endif
