/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

static constexpr auto time_scale        = 0.1;   // slow down time
static constexpr auto count             = 40UL;  // hiw many terms of Fourier series
static constexpr auto integration_steps = 100UL; // how many steps to use while integrating

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

using complex = std::complex<f64>;

// parametric square shape
auto target_shape(f64 t)
{
    auto out = complex{};
    if (t < 0.25)
    {
        out = interp::map_range<f64, complex>(
            t, Extents<f64>{0.0, 0.25}, Extents<complex>{complex{1.0, -1.0}, complex{1.0, 1.0}});
    }
    else if (t < 0.5)
    {
        out = interp::map_range<f64, complex>(
            t, Extents<f64>{0.25, 0.5}, Extents<complex>{complex{1.0, 1.0}, complex{-1.0, 1.0}});
    }
    else if (t < 0.75)
    {
        out = interp::map_range<f64, complex>(
            t, Extents<f64>{0.5, 0.75}, Extents<complex>{complex{-1.0, 1.0}, complex{-1.0, -1.0}});
    }
    else
    {
        out = interp::map_range<f64, complex>(
            t, Extents<f64>{0.75, 1.0}, Extents<complex>{complex{-1.0, -1.0}, complex{1.0, -1.0}});
    }
    return to_complex(from_complex(out).rotated_by(1) * Vector2{1.0, 1.6}) +
           complex{0.3, -0.4}; // rotate, scale to make it interesting
}

auto raise_to_power(complex x) { return std::pow(math::e, math::two_pi_i * x); }

int main()
{
    constexpr auto indices =
        irange(-i32(count) / 2, i32(count) / 2); // indices of terms of Foruier series


    //!!!!!!!!!
    // Where the interesting stuff happens:
    auto coefficients = std::vector<complex>(count, {1.0});

    for (auto i : indices)
    {
        const auto fn = [&](f64 t) { return raise_to_power(-f64(i) * t) * target_shape(t); };

        coefficients[u64(i) + count / 2UL] = // remap i as i is in range[-count / 2, count / 2)
            math::integral<complex, f64>(fn, 0.0, 1.0, integration_steps);
    }
    //!!!!!!!!!

    auto app   = App{{.dims = {1000, 1000}}};
    auto watch = Stopwatch{};
    auto trail = Trail{300}; // trail of pen

    const auto draw = [&]
    {
        app.fill("#06060c"_c);

        const auto time = watch.time().count() * time_scale;

        auto sum = complex{};
        for (auto i : indices)
        {
            auto current = raise_to_power(f64(i) * time);  // base vector
            current *= coefficients[u64(i) + count / 2UL]; // apply coefficient

            app.draw_line_segment({from_complex(sum), from_complex(sum + current)}, colors::white,
                                  0.008);
            sum += current;
        }
        trail.push_back(from_complex(sum)); // add current position to trail

        // app.draw_polyline(math::sample<f64, complex>(target_shape, 0.0, 1.0, integration_steps),
        //                   colors::antiquewhite, 0.03);


        // Draw trail:
        for (auto i : range(trail.size() - 1UL))
        {
            app.draw_line_segment({trail[i], trail[i + 1UL]},
                                  ("#4d83f7"_c).with_multiplied_alpha(f64(i) / f64(trail.size())),
                                  0.012);
        }
    };

    app.transform.scale *= 20; // zoom in
    app.run(draw);
}
