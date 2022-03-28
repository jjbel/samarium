/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "fmt/format.h"

#include "Stopwatch.hpp"

namespace sm::util
{
void Stopwatch::reset() { start = std::chrono::steady_clock::now(); }

Stopwatch::Duration_t Stopwatch::time() const
{
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration_t>(finish - start);
}

void Stopwatch::print() const { fmt::print("Took {:.3}ms\n", this->time().count() * 1000); }

}; // namespace sm::util
