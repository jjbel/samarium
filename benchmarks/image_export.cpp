/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/util/file.hpp"

using namespace sm;

static void bm_file_export_Targa(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::export_to(file::Targa{}, image, "benchmark.tga"); }
    std::filesystem::remove("benchmark.tga");
}

static void bm_file_export_Bmp(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::export_to(file::Bmp{}, image, "benchmark.bmp"); }
    std::filesystem::remove("benchmark.bmp");
}

static void bm_file_export_Png(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::export_to(file::Bmp{}, image, "benchmark.png"); }
    std::filesystem::remove("benchmark.png");
}

BENCHMARK(bm_file_export_Targa)
    ->Name("file::export_to(file::Targa)")
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);

BENCHMARK(bm_file_export_Bmp)
    ->Name("file::export_to(file::Bmp)")
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);

BENCHMARK(bm_file_export_Png)
    ->Name("file::export_to(file::Png)")
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);
