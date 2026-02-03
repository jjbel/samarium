/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/math.hpp"
#include "samarium/util/FunctionRef.hpp"

namespace sm
{
using complex = std::complex<f64>;
using ShapeFn = FunctionRef<complex(f64)>;

// indices of terms of Fourier series
// indices go [0, -1, 1, -2, 2...]
auto make_indices(u64 count)
{
    auto indices = std::vector<i32>(count);

    for (auto i : loop::end(count))
    {
        if (i % 2 == 0) { indices[i] = i32(i) / 2; }
        else { indices[i] = -(i32(i) + 1) / 2; }
    }
    return indices;
}

auto raise_to_power(complex x) { return std::pow(math::e, math::two_pi_i * x); }

// does the actual fourier transform
auto coefficients(ShapeFn shape, u64 count, u64 integration_steps)
{
    const auto indices = make_indices(count);
    auto coefficients  = std::vector<complex>(count, {1.0});

    // TODO split this into a different function for just 1 coeff
    // add a way to pass any indices, have a mapping from frequencies to array indices

    for (auto i : indices)
    {
        const auto fn = [&](f64 t) { return raise_to_power(-f64(i) * t) * shape(t); };

        coefficients[u64(i) + count / 2UL] = // remap i as i is in range[-count / 2, count / 2)
            math::integral<complex, f64>(fn, 0.0, 1.0, integration_steps);
    }
    return coefficients;
}

auto shape_vertices(ShapeFn shape, u64 integration_steps)
{
    auto vec = std::vector<Vec2>();
    vec.reserve(integration_steps);
    for (auto i : math::sample<f64, complex>(shape, 0.0, 1.0, integration_steps))
    {
        vec.push_back(from_complex(i));
    }
    return vec;
}

struct ShapeFnFromPts
{
    std::span<const Vec2> points;

    auto operator()(f64 t) const
    {
        // TODO bad of t=1?
        t *= points.size() - 1;
        const auto index  = static_cast<u64>(std::floor(t));
        const auto factor = math::mod(t, 1.0);
        return to_complex(interp::lerp<Vec2>(factor, {points[index], points[index]}));
    }
};

// simple square shape
// scaled and rotated
auto square(f64 t)
{
    auto mapper = [t](f64 a, f64 b, f64 c, f64 d, f64 e, f64 f)
    {
        return interp::map_range<f64, complex>(t, Extents<f64>{a, b},
                                               Extents<complex>{complex{c, d}, complex{e, f}});
    };

    auto out = complex{};
    if (t < 0.25) { out = mapper(0.0, 0.25, 1.0, -1.0, 1.0, 1.0); }
    else if (t < 0.5) { out = mapper(0.25, 0.5, 1.0, 1.0, -1.0, 1.0); }
    else if (t < 0.75) { out = mapper(0.5, 0.75, -1.0, 1.0, -1.0, -1.0); }
    else { out = mapper(0.75, 1.0, -1.0, -1.0, 1.0, -1.0); }

    return to_complex(from_complex(out).rotated(1) * Vec2{1.0, 1.6} + Vec2{0.0, 0.5}) +
           complex{0.3, -0.4}; // rotate, scale to make it interesting
}
} // namespace sm
