/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */


#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "fourier.hpp"
#include "parse_obj.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    constexpr auto time_scale          = .4;    // speed up time
    constexpr auto integration_steps   = 100UL; // how many steps to use while integrating
    constexpr auto fourier_terms_count = 140UL;  // how many terms of Fourier series
    constexpr auto trail_size          = 44000UL;

    auto level      = 100; // what level to start drawing at
    auto level_step = 10;  // change of level on pressing Up or Down arrow

    // auto points = obj_to_pts(".\\examples\\fourier\\india.obj");
    auto points = obj_to_pts("D:\\blender\\2024-10-23 fourier card\\03.obj");

    // sometimes u need to flip the verts coz blener has different coord sys
    // for (auto& point : points) { point.y *= -1; }

    const auto shape = ShapeFnFromPts{points};

    // or try
    // const auto shape = square;

    // TODO integration_steps = 300 gives spikes
    // or increasing fourier_terms_count
    // sometimes decreasing timescale helps
    // maybe fourier correct but integration when finding trail wrong?

    // TODO zooming now working but pan(), zoom() works
    constexpr auto zoom_out          = 2.7; // zoom out
    constexpr auto dims              = Dimensions{1080, 1080};
    constexpr auto trail_color       = "#ff0850"_c;
    constexpr auto trail_thickness   = 0.036F;
    constexpr auto vectors_color     = "#08b8ff"_c;
    constexpr auto vectors_thickness = 0.023F;
    constexpr auto background_color  = "#080819"_c;

    // --------------------------------------------------------------------

    const auto indices = make_indices(fourier_terms_count);
    const auto coeffs  = coefficients(shape, fourier_terms_count, integration_steps);

    auto window = Window{{.dims = dims}};
    auto watch  = Stopwatch{};
    auto trail  = Trail{trail_size}; // trail of pen


    auto level_up =
        keyboard::OnKeyDown{*window.handle,
                            {Key::Up},
                            [&]
                            {
                                level += level_step;
                                level = std::min(level, static_cast<i32>(fourier_terms_count));
                                print("Level:", level);
                            }};

    auto level_down = keyboard::OnKeyDown{*window.handle,
                                          {Key::Down},
                                          [&]
                                          {
                                              level -= level_step;
                                              level = std::max(level, 1);
                                              print("Level:", level);
                                          }};

    auto frame = 0;

    const auto draw_trail = [&]
    {
        if (trail.size() >= 2)
        {
            //     draw::polyline_segments(window, points_to_f32(trail.trail), 0.08F, "#FF1C58"_c);
            for (auto i : loop::end(trail.size() - 1UL))
            {
                draw::line_segment(window, {trail[i], trail[i + 1UL]},
                                   trail_color.with_multiplied_alpha(
                                       interp::ease_out_quint(f64(i) / f64(trail.size()))),
                                   trail_thickness);
            }
        }
    };

    const auto draw_vectors = [&]
    {
        const auto time = frame / 2000.0 * time_scale;
        auto sum        = complex{};

        // TODO use enumerate
        auto j = 0;
        for (auto i : indices)
        {
            const auto index = u64(i) + fourier_terms_count / 2UL;

            if (j > level) { break; }
            j++;

            auto current = raise_to_power(f64(i) * time); // base vector
            current *= coeffs[index];                     // apply coefficient

            draw::line_segment(window, LineSegment{from_complex(sum), from_complex(sum + current)},
                               vectors_color, vectors_thickness /* / f32(j) */);

            // draw::circle(window, Circle{from_complex(sum), 0.05},
            //              colors::white);

            sum += current;
        }

        trail.push_back(from_complex(sum)); // add current position to trail
    };

    const auto draw = [&]
    {
        draw::background(background_color);
        level_up();
        level_down();

        // TODO draw vectors also contains building the trail code, so doesn't make sense to put in
        // a separate fn use flags instead
        draw_trail();
        draw_vectors();
        // make path and steps CLI params

        // draw the original shape:
        // draw::polygon_segments(window, points_to_f32(points), 0.02F, ("#4d83f7"_c));

        window.pan();
        window.zoom_to_cursor();
        frame++;

        // export to images:
        file::write(file::pam, window.get_image(), fmt::format("./exports/{:05}.pam", frame));
    };

    window.camera.scale /= zoom_out;
    run(window, draw);
    print(frame / watch.seconds(), "fps");
}
