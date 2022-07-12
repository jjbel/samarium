/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <iostream>

// #include "Eigen/Core"
#include "benchmark/benchmark.h"

#include "samarium/physics/ParticleSystem.hpp"
#include "samarium/util/RandomGenerator.hpp"

// using Array = Eigen::Array<double, Eigen::Dynamic, 1>;

using namespace sm;

static void bm_ParticleSystem_update(benchmark::State& state)
{
    auto rand = RandomGenerator{};
    // Array array(20);
    // std::cout << array;
    auto ps = ParticleSystem{100};
    for (auto& p : ps) { p.acc = rand.polar_vector({0.0, 12.0}); }

    for (auto _ : state) { ps.update(); }
}

static void bm_ParticleSystem_update_self_collision(benchmark::State& state)
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

BENCHMARK(bm_ParticleSystem_update)->Name("ParticleSystem::update()");
BENCHMARK(bm_ParticleSystem_update_self_collision)->Name("Particlesystem::self_collision()");
