/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

static constexpr auto time_scale   = 0.03;
static constexpr auto count        = 40UL;
static constexpr auto sample_steps = 40UL;

using namespace sm;
using namespace sm::literals;

using complex = std::complex<f64>;

int main()
{
    const auto two_pi_i    = 2.0 * math::pi * complex{0.0, 1.0};
    constexpr auto indices = irange(-static_cast<i32>(count) / 2, static_cast<i32>(count) / 2);

    auto coefficients = std::vector<complex>(count, {1.0});

    const auto target_shape = [](f64 t)
    {
        auto out = complex{};
        if (t < 0.25)
        {
            out = interp::map_range<f64, complex>(
                t, Extents<f64>{0.0, 0.25},
                Extents<complex>{complex{1.0, -1.0}, complex{1.0, 1.0}});
        }
        else if (t < 0.5)
        {
            out = interp::map_range<f64, complex>(
                t, Extents<f64>{0.25, 0.5},
                Extents<complex>{complex{1.0, 1.0}, complex{-1.0, 1.0}});
        }
        else if (t < 0.75)
        {
            out = interp::map_range<f64, complex>(
                t, Extents<f64>{0.5, 0.75},
                Extents<complex>{complex{-1.0, 1.0}, complex{-1.0, -1.0}});
        }
        else
        {
            out = interp::map_range<f64, complex>(
                t, Extents<f64>{0.75, 1.0},
                Extents<complex>{complex{-1.0, -1.0}, complex{1.0, -1.0}});
        }
        return out + complex{0.3, -0.4};
    };

    /////////////////////////
    // Where the interesting stuff happens:

    for (auto i : indices)
    {
        const auto fn = [&](f64 t)
        { return std::pow(math::e, -two_pi_i * static_cast<f64>(i) * t) * target_shape(t); };

        coefficients[u64(i) + count / 2UL] =
            math::integral<complex, f64>(fn, 0.0, 1.0, sample_steps);
    }
    /////////////////////////

    auto app   = App{{.dims = {1000, 1000}}};
    auto watch = Stopwatch{};
    auto trail = Trail{100};

    const auto draw = [&]
    {
        app.fill("#06060c"_c);

        const auto time = watch.time().count() * math::two_pi * time_scale;

        auto sum = complex{};
        for (auto i : indices)
        {
            const auto current =
                coefficients[u64(i) + count / 2UL] *
                std::pow(math::e, math::two_pi * complex{0.0, 1.0} * f64(i) * time);

            app.draw_line_segment(
                LineSegment{from_complex(sum), from_complex(sum + current * 0.96)}, colors::white,
                0.01);
            sum += current;
        }
        trail.push_back(from_complex(sum));

        for (auto i : math::sample<f64, complex>(target_shape, 0.0, 1.0, sample_steps))
        {
            app.draw(Circle{from_complex(i), 0.01}, colors::azure);
        }


        for (auto i : range(trail.size() - 1UL))
        {
            app.draw_line_segment({trail[i], trail[i + 1UL]},
                                  colors::yellow.with_multiplied_alpha(f64(i) / f64(trail.size())),
                                  0.008);
        }
    };

    app.transform.scale *= 20; // zoom in
    app.run(draw);
}
