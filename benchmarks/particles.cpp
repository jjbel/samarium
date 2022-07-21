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

static void bm_ParticleSystem_update(benchmark::State& state)
{
    auto rand = RandomGenerator{};
    auto ps   = ParticleSystem{100};
    for (auto& p : ps) { p.acc = rand.polar_vector({0.0, 12.0}); }

    for (auto _ : state) { ps.update(); }
}

static void bm_ParticleSystem_update_self_collision(benchmark::State& state)
{
    auto rand        = RandomGenerator{};
    auto ps          = ParticleSystem{100};
    const auto width = 20;
    for (auto& p : ps)
    {
        p.pos    = rand.vector({{-width, -width}, {width, width}});
        p.acc    = rand.polar_vector({0.0, 12.0});
        p.radius = 0.5;
    }

    for (auto _ : state)
    {
        ps.self_collision();
        ps.update();
    }
}

static void bm_ParticleSystem_update_self_collision_with_threshold(benchmark::State& state)
{
    auto rand                     = RandomGenerator{};
    auto ps                       = ParticleSystem{100};
    const auto width              = 20;
    const auto distance_threshold = static_cast<f64>(state.range(0));
    for (auto& p : ps)
    {
        p.pos    = rand.vector({{-width, -width}, {width, width}});
        p.acc    = rand.polar_vector({0.0, 12.0});
        p.radius = 0.5;
    }

    for (auto _ : state)
    {
        ps.self_collision(1.0, distance_threshold);
        ps.update();
    }
}

BENCHMARK(bm_ParticleSystem_update)->Name("ParticleSystem::update()");
BENCHMARK(bm_ParticleSystem_update_self_collision)->Name("Particlesystem::self_collision()");
BENCHMARK(bm_ParticleSystem_update_self_collision_with_threshold)
    ->Name("Particlesystem::self_collision() with threshold")
    ->DenseRange(1, 40, 8);
