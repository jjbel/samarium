/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>

#include "samarium/core/types.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/loop.hpp" // for end

namespace sm::noise
{
struct Params
{
    f64 scale{1.0};
    u64 detail{4};
    f64 seed{0.0};
    f64 roughness{0.5};
};

constexpr inline auto hash_array = std::array<u8, 256>{
    {208, 34,  231, 213, 32,  248, 233, 56,  161, 78,  24,  140, 71,  48,  140, 254, 245, 255, 247,
     247, 40,  185, 248, 251, 245, 28,  124, 204, 204, 76,  36,  1,   107, 28,  234, 163, 202, 224,
     245, 128, 167, 204, 9,   92,  217, 54,  239, 174, 173, 102, 193, 189, 190, 121, 100, 108, 167,
     44,  43,  77,  180, 204, 8,   81,  70,  223, 11,  38,  24,  254, 210, 210, 177, 32,  81,  195,
     243, 125, 8,   169, 112, 32,  97,  53,  195, 13,  203, 9,   47,  104, 125, 117, 114, 124, 165,
     203, 181, 235, 193, 206, 70,  180, 174, 0,   167, 181, 41,  164, 30,  116, 127, 198, 245, 146,
     87,  224, 149, 206, 57,  4,   192, 210, 65,  210, 129, 240, 178, 105, 228, 108, 245, 148, 140,
     40,  35,  195, 38,  58,  65,  207, 215, 253, 65,  85,  208, 76,  62,  3,   237, 55,  89,  232,
     50,  217, 64,  244, 157, 199, 121, 252, 90,  17,  212, 203, 149, 152, 140, 187, 234, 177, 73,
     174, 193, 100, 192, 143, 97,  53,  145, 135, 19,  103, 13,  90,  135, 151, 199, 91,  239, 247,
     33,  39,  145, 101, 120, 99,  3,   186, 86,  99,  41,  237, 203, 111, 79,  220, 135, 158, 42,
     30,  154, 120, 67,  87,  167, 135, 176, 183, 191, 253, 115, 184, 21,  233, 58,  129, 233, 142,
     39,  128, 211, 118, 137, 139, 255, 114, 20,  218, 113, 154, 27,  127, 246, 250, 1,   8,   198,
     250, 209, 92,  222, 173, 21,  88,  102, 219}};

auto perlin_2d(Vector2 pos, Params params = {}) -> f64;

auto perlin_1d(f64 pos, Params params = {}) -> f64;
} // namespace sm::noise


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_NOISE_IMPL)

#include <cmath>

#include "samarium/math/Extents.hpp"
#include "samarium/math/interp.hpp"

namespace sm::noise
{
// from https://gist.github.com/nowl/828013

auto interp_smooth(f64 a, f64 b, f64 s)
{
    return interp::lerp<f64>(s * s * (3.0 - 2.0 * s), {a, b});
}

auto noise1d(i32 x) -> i32
{
    constexpr auto yindex = 69UL;
    auto xindex           = (hash_array[yindex] + x) % 256;
    if (xindex < 0) { xindex += 256; }
    return hash_array[static_cast<u64>(xindex)];
}

auto single_iter_noise1d(f64 pos) -> f64
{
    const auto x_int  = static_cast<i32>(std::floor(pos));
    const auto x_frac = pos - static_cast<f64>(x_int);
    const auto s      = noise1d(x_int);
    const auto t      = noise1d(x_int + 1);
    return interp_smooth(s, t, x_frac);
}

auto noise2d(i32 x, i32 y)
{
    constexpr auto seed = 2022;
    auto yindex         = (y + seed) % 256;
    if (yindex < 0) { yindex += 256; }
    auto xindex = (hash_array[static_cast<u64>(yindex)] + x) % 256;
    if (xindex < 0) { xindex += 256; }
    return hash_array[static_cast<u64>(xindex)];
}

auto single_iter_noise2d(Vector2 pos)
{
    const auto x_int  = static_cast<i32>(std::floor(pos.x));
    const auto y_int  = static_cast<i32>(std::floor(pos.y));
    const auto x_frac = pos.x - static_cast<f64>(x_int);
    const auto y_frac = pos.y - static_cast<f64>(y_int);
    const auto s      = noise2d(x_int, y_int);
    const auto t      = noise2d(x_int + 1, y_int);
    const auto u      = noise2d(x_int, y_int + 1);
    const auto v      = noise2d(x_int + 1, y_int + 1);
    const auto low    = interp_smooth(s, t, x_frac);
    const auto high   = interp_smooth(u, v, x_frac);
    return interp_smooth(low, high, y_frac);
}

auto perlin_1d(f64 pos, Params params) -> f64
{
    pos            = (pos + 100.0 + 200.0 * params.seed) * params.scale / 10.0;
    auto amplitude = 1.0;
    auto result    = 0.0;
    auto div       = 0.0;

    for (auto i : loop::end(params.detail))
    {
        div += 256 * amplitude;
        result += single_iter_noise1d(pos) * amplitude;
        amplitude *= params.roughness;
        pos *= 2.0;

        std::ignore = i;
    }
    return result / div;
}

auto perlin_2d(Vector2 pos, Params params) -> f64
{
    pos = (pos + Vector2{100, 100} + Vector2{200, 200} * params.seed) * params.scale / 10.0;
    auto amplitude = 1.0;
    auto result    = 0.0;
    auto div       = 0.0;

    for (auto i : loop::end(params.detail))
    {
        div += 256.0 * amplitude;
        result += single_iter_noise2d(pos) * amplitude;
        amplitude *= params.roughness;
        pos *= 2.0;

        std::ignore = i;
    }
    return result / div;
}
} // namespace sm::noise
#endif
