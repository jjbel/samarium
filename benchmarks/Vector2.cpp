/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/math/Vector2.hpp"
#include "samarium/math/math.hpp"
#include "samarium/util/RandomGenerator.hpp"

using namespace sm;

static void Vector2_length(benchmark::State& state)
{
    auto vec  = Vector2{1.0, 0.0};
    auto rand = RandomGenerator{};

    for (auto _ : state) { benchmark::DoNotOptimize(vec.length()); }
    state.SetItemsProcessed(state.iterations());
}

static void Vector2_angle(benchmark::State& state)
{
    auto vec  = Vector2{1.0, 0.0};
    auto rand = RandomGenerator{};

    for (auto _ : state) { benchmark::DoNotOptimize(vec.angle()); }
    state.SetItemsProcessed(state.iterations());
}

static void Vector2_rotate(benchmark::State& state)
{
    auto vec  = Vector2{1.0, 0.0};
    auto rand = RandomGenerator{};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize((vec.rotate(rand.range<f64>({0.0, math::two_pi})), 1));
    }
    state.SetItemsProcessed(state.iterations());
}

static void Vector2_set_angle(benchmark::State& state)
{
    auto vec  = Vector2{1.0, 0.0};
    auto rand = RandomGenerator{};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize((vec.set_angle(rand.range<f64>({0.0, math::two_pi})), 1));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Vector2_length);
BENCHMARK(Vector2_angle);
BENCHMARK(Vector2_rotate);
BENCHMARK(Vector2_set_angle);
