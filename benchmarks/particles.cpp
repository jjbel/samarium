/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/physics/ParticleSystem.hpp"
#include "samarium/util/RandomGenerator.hpp"

using namespace sm;

static void BM_ParticleSystem(benchmark::State& state)
{
    auto rand = RandomGenerator{};
    auto ps   = ParticleSystem{100};
    for (auto& p : ps) { p.acc = rand.polar_vector({0.0, 12.0}); }

    for (auto _ : state)
    {
        ps.self_collision();
        ps.update();
    }
}

BENCHMARK(BM_ParticleSystem);
