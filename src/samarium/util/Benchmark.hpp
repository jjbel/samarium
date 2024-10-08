/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <chrono> // for duration
#include <string> // for string
#include <vector> // for vector


#include "samarium/core/types.hpp"     // for Float
#include "samarium/util/unordered.hpp" // for Map

namespace sm
{


struct Benchmark
{
    using Float    = f32;
    using Duration = std::chrono::duration<Float>;
    using Clock    = std::chrono::steady_clock;

    struct Stats
    {
        Float mean{};
        Float median{};
        Float min{};
        Float max{};
        Float stddev{};
    };

    const Clock::time_point start{Clock::now()};
    Clock::time_point now{Clock::now()};
    Clock::time_point frame_start{Clock::now()};
    u64 frame_count{};
    Map<std::string, std::vector<Float>> times;

    void reset();

    void add(const std::string& key);

    void add_frame();

    [[nodiscard]] auto time() const -> Duration;

    [[nodiscard]] auto seconds() const -> Float;

    void print();

  private:
    [[nodiscard]] auto get_stats(std::vector<Float>& data) -> Stats;

    [[nodiscard]] auto get_max_key_size() -> u64;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_BENCHMARK_IMPL)

#include <ranges> // for transform

#include "fmt/color.h"
#include "range/v3/action/sort.hpp"           // for sort
#include "range/v3/algorithm/max_element.hpp" // for max_element

#include "samarium/core/inline.hpp" // for SM_INLINE
#include "samarium/math/stats.hpp"  // for sum, mean


namespace sm
{
SM_INLINE void Benchmark::add(const std::string& key)
{
    const auto finish = Clock::now();
    // store millis!
    times[key].push_back(std::chrono::duration_cast<Duration>(finish - now).count() * 1000.0F);
    now = finish; // reset
}

SM_INLINE void Benchmark::add_frame()
{
    const auto finish = Clock::now();
    times["frame"].push_back(std::chrono::duration_cast<Duration>(finish - frame_start).count() *
                             1000.0F);
    frame_start = finish; // reset
    frame_count++;
}


SM_INLINE void Benchmark::reset() { now = Clock::now(); }

SM_INLINE auto Benchmark::time() const -> Benchmark::Duration
{
    const auto finish = Clock::now();
    return std::chrono::duration_cast<Duration>(finish - now);
}

[[nodiscard]] SM_INLINE auto Benchmark::seconds() const -> Float { return this->time().count(); }

[[nodiscard]] SM_INLINE auto Benchmark::get_stats(std::vector<Float>& data) -> Stats
{
    ranges::actions::sort(data);
    const auto mean = math::mean<Float>(data);
    auto squares    = data;
    for (auto& i : squares)
    {
        const auto error = (i - mean);
        i                = error * error;
    }

    return Stats{.mean   = mean,
                 .median = data[data.size() / 2], // span doesn't have at() yet
                 .min    = data[0],
                 .max    = data[data.size() - 1],
                 .stddev = std::sqrt(math::mean<Float>(squares))};
}

[[nodiscard]] SM_INLINE auto Benchmark::get_max_key_size() -> u64
{
    auto max_key_size = u64{};

    for (const auto& [key, data] : times)
    {
        if (key.size() > max_key_size) { max_key_size = key.size(); }
    }
    return max_key_size;
}

SM_INLINE void Benchmark::print()
{
    const auto seconds_since_start = std::chrono::duration_cast<Duration>(Clock::now() - start);
    // fmt::print("frametime: {: >10.5}ms\n", seconds() * 1000.0);
    fmt::print(fmt::emphasis::bold, "fps: {: >4.1f}\n",
               static_cast<Float>(frame_count) / seconds_since_start.count());


    // TODO gives ranges dangling. print typeid(max_key_size).name()
    // const auto max_key_size = ranges::max_element(
    //    times | std::views::transform([](const auto& pair) { return pair.first.size(); }));

    const auto max_key_size = get_max_key_size();
    fmt::print(fmt::emphasis::bold, "{:<{}} ", "Task", max_key_size);
    for (const auto& heading : {"mean", "median", "min", "max", "stddev"})
    {
        fmt::print(fmt::emphasis::bold, "{: >6} ", heading);
    }
    fmt::print("\n");

    for (auto& [key, data] : times)
    {
        const auto stats = get_stats(data);
        fmt::print("{:<{}} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}\n", key, max_key_size,
                   stats.mean, stats.median, stats.min, stats.max, stats.stddev);
    }
}

}; // namespace sm
#endif
