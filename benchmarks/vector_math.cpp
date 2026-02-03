/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/math/math.hpp"
#include "samarium/math/vector_math.hpp"
#include "samarium/util/RandomGenerator.hpp"

using namespace sm;

/*
    regular_poly_f32 : seems to be a lot of variation on re-running, esp for tri
    also f64 seems to be faster than f32, or roughly the same
*/

static void regular_poly_f32(benchmark::State& state)
{
    auto points = static_cast<u32>(state.range(0));
    auto circle = Circle{{0, 0}, 1};
    auto rand   = RandomGenerator{}; // TODO rand why

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(math::regular_polygon_points<f32>(points, circle));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(regular_poly_f32)->Arg(3)->Arg(4)->Arg(8)->Arg(12)->Arg(16)->Arg(32)->Arg(48)->Arg(64);

static void regular_poly_f64(benchmark::State& state)
{
    auto points = static_cast<u32>(state.range(0));
    auto circle = Circle{{0, 0}, 1};
    auto rand   = RandomGenerator{}; // TODO rand why

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(math::regular_polygon_points<f64>(points, circle));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(regular_poly_f64)->Arg(3)->Arg(4)->Arg(8)->Arg(12)->Arg(16)->Arg(32)->Arg(48)->Arg(64);
