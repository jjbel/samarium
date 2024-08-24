/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "benchmark/benchmark.h"

#include "samarium/util/file.hpp"

using namespace sm;

static void bm_file_export_Targa(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::write(file::Targa{}, image, "benchmark.tga"); }
    std::filesystem::remove("benchmark.tga");
    state.SetItemsProcessed(state.iterations());
}

static void bm_file_export_Pam(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::write(file::Pam{}, image, "benchmark.pam"); }
    std::filesystem::remove("benchmark.pam");
    state.SetItemsProcessed(state.iterations());
}

static void bm_file_export_Bmp(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::write(file::Bmp{}, image, "benchmark.bmp"); }
    std::filesystem::remove("benchmark.bmp");
    state.SetItemsProcessed(state.iterations());
}

static void bm_file_export_Png(benchmark::State& state)
{
    const auto image = Image{{static_cast<u64>(state.range(0)), static_cast<u64>(state.range(0))}};
    for (auto _ : state) { file::write(file::Bmp{}, image, "benchmark.png"); }
    std::filesystem::remove("benchmark.png");
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(bm_file_export_Pam)
    ->Name("Pam")
    ->Unit(benchmark::kMillisecond)
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);

BENCHMARK(bm_file_export_Targa)
    ->Name("Targa")
    ->Unit(benchmark::kMillisecond)
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);

BENCHMARK(bm_file_export_Bmp)
    ->Name("Bmp")
    ->Unit(benchmark::kMillisecond)
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);

BENCHMARK(bm_file_export_Png)
    ->Name("Png")
    ->Unit(benchmark::kMillisecond)
    ->Arg(200)
    ->Arg(800)
    ->Arg(1600)
    ->Arg(3200);
