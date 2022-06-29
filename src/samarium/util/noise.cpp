/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <cmath>

#include "samarium/math/Extents.hpp"
#include "samarium/math/interp.hpp"

#include "noise.hpp"

namespace sm::noise
{
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
    pos            = (pos + 100.0) * params.scale / 10.0;
    auto amplitude = 1.0;
    auto result    = 0.0;
    auto div       = 0.0;

    for (auto i : range(params.detail))
    {
        div += 256 * amplitude;
        result += single_iter_noise1d(pos) * amplitude;
        amplitude *= params.roughness;
        pos *= 2.0;
    }
    return result / div;
}

auto perlin_2d(Vector2 pos, Params params) -> f64
{
    pos            = (pos + Vector2{100, 100}) * params.scale / 10.0;
    auto amplitude = 1.0;
    auto result    = 0.0;
    auto div       = 0.0;

    for (auto i : range(params.detail))
    {
        div += 256.0 * amplitude;
        result += single_iter_noise2d(pos) * amplitude;
        amplitude *= params.roughness;
        pos *= 2.0;
    }
    return result / div;
}
} // namespace sm::noise
