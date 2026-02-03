/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/util/RandomGenerator.hpp"
#include "samarium/util/print.hpp"

using namespace sm;

static constexpr u64 cache_size = 1'000'000;

static void bm_RandomGenerator_uncached(benchmark::State& state)
{
    auto rand = RandomGenerator{0};

    for (auto _ : state) { benchmark::DoNotOptimize(rand.random()); }
    state.SetItemsProcessed(state.iterations());
}

static void bm_RandomGenerator_cached(benchmark::State& state)
{
    auto rand = RandomGenerator{cache_size};
    for (auto _ : state) { benchmark::DoNotOptimize(rand.random()); }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(bm_RandomGenerator_uncached)
    ->Name("RandomGenerator::random() without cache")
    ->Iterations(cache_size);
BENCHMARK(bm_RandomGenerator_cached)
    ->Name("RandomGenerator::random() with cache")
    ->Iterations(cache_size);
