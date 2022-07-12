/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/util/RandomGenerator.hpp"
#include "samarium/util/print.hpp"

using namespace sm;

static void bm_RandomGenerator_uncached(benchmark::State& state)
{
    auto rand = RandomGenerator{0};

    for (auto _ : state) { benchmark::DoNotOptimize(rand.random()); }
}

static void bm_RandomGenerator_cached(benchmark::State& state)
{
    auto rand = RandomGenerator(1024);

    for (auto _ : state) { benchmark::DoNotOptimize(rand.random()); }
}

BENCHMARK(bm_RandomGenerator_uncached)->Name("RandomGenerator::random() without cache");
BENCHMARK(bm_RandomGenerator_cached)->Name("RandomGenerator::random() with cache");
