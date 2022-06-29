/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <cmath>
#include <random>

#include "samarium/math/Extents.hpp"
#include "samarium/util/print.hpp"

#include "range/v3/algorithm/shuffle.hpp"
#include "range/v3/numeric/iota.hpp"

#include "PerlinNoise.hpp"

namespace sm::util
{
f64 fade(f64 t) { return t * t * t * (t * (t * 6 - 15) + 10); }

f64 lerp(f64 t, f64 a, f64 b) { return a + t * (b - a); }

f64 grad(i32 hash, f64 x, f64 y, f64 z)
{
    const i32 h = hash & 15;
    // Convert lower 4 bits of hash into 12 gradient directions
    const f64 u = h < 8 ? x : y, v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

PerlinNoise::PerlinNoise()
{
    // Initialize the permutation vector with the reference values
    p = std::vector(default_permutation.begin(), default_permutation.end());
    // Duplicate the permutation vector
    p.insert(p.end(), p.begin(), p.end());
}

// Generate a new permutation vector based on the value of seed
PerlinNoise::PerlinNoise(u32 seed)
{
    p.resize(256);

    // Fill p with values from 0 to 255
    ranges::iota(p, 0);

    // Initialize a random engine with seed
    auto engine = std::default_random_engine(seed);

    // Suffle  using the above random engine
    ranges::shuffle(p, engine);

    // Duplicate the permutation vector
    p.insert(p.end(), p.begin(), p.end());
}

auto PerlinNoise::noise(f64 x, f64 y, f64 z) const -> f64
{
    // Find the unit cube that contains the point
    const auto X = i32(std::floor(x)) & 255;
    const auto Y = i32(std::floor(y)) & 255;
    const auto Z = i32(std::floor(z)) & 255;

    // Find relative x, y,z of point in cube
    x -= std::floor(x);
    y -= std::floor(y);
    z -= std::floor(z);

    // Compute fade curves for each of x, y, z
    const auto u = fade(x);
    const auto v = fade(y);
    const auto w = fade(z);

    // Hash coordinates of the 8 cube corners
    const auto A  = p[u64(X)] + Y;
    const auto AA = p[u64(A)] + Z;
    const auto AB = p[u64(A + 1)] + Z;
    const auto B  = p[u64(X + 1)] + Y;
    const auto BA = p[u64(B)] + Z;
    const auto BB = p[u64(B + 1)] + Z;

    // Add blended results from 8 corners of cube
    const auto result = lerp(
        w,
        lerp(v, lerp(u, grad(p[u64(AA)], x, y, z), grad(p[u64(BA)], x - 1, y, z)),
             lerp(u, grad(p[u64(AB)], x, y - 1, z), grad(p[u64(BB)], x - 1, y - 1, z))),
        lerp(v, lerp(u, grad(p[u64(AA + 1)], x, y, z - 1), grad(p[u64(BA + 1)], x - 1, y, z - 1)),
             lerp(u, grad(p[u64(AB + 1)], x, y - 1, z - 1),
                  grad(p[u64(BB + 1)], x - 1, y - 1, z - 1))));
    return result;
}

auto PerlinNoise::operator()(f64 x, f64 y, f64 z) const -> f64 { return noise(x, y, z); }

auto PerlinNoise::detail(Vector2 position, NoiseParams params) const -> f64
{
    position *= params.scale;
    auto result = 0.0;
    auto factor = 0.0;

    auto scale = 1.0;

    for (auto i : range(params.detail))
    {
        result += scale * this->operator()(position.x, position.y);
        position /= scale;
        factor += scale;
        scale *= 0.5;
    }

    // print(result, factor, result / factor);

    return (result + factor) / (2 * factor);
}
} // namespace sm::util
