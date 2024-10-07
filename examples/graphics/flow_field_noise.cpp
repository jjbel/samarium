/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"
#include "range/v3/view/enumerate.hpp"

using namespace sm;
using namespace sm::literals;

constexpr auto window_dims     = Dimensions{1000, 1000};
constexpr u64 downscale_factor = 20;
constexpr u64 particle_count   = 100000;
constexpr f64 radius           = 0.0045;
constexpr f64 transparency     = 1.0;
constexpr f64 sim_speed        = .125;
constexpr f64 max_speed        = 4;
constexpr f64 max_force        = 1.0;
constexpr f64 mouse_dist       = 0.05;

auto main() -> i32
{
    // TODO also see old values (the ones in 1920x1080). gave better results
    auto window        = Window{{.dims = window_dims}};
    auto watch         = Stopwatch{};
    auto frame_counter = 0;

    const auto scale = .4;
    const auto dims  = window.dims / downscale_factor; // make this window dims / detail

    auto ps     = ParticleSystem{particle_count};
    auto forces = VectorField{dims};
    auto rand   = RandomGenerator{1000'000, RandomMode::Stable, 42};

    const auto dims_f64 = dims.cast<f64>();

    // instead of doing any mapping, should change the window view
    // TODO VVIMP mapping pixel space to graph space.
    const auto aspect_ratio_reciprocal = window.dims.cast<f64>().slope();
    const auto field_to_graph          = [dims_f64, aspect_ratio_reciprocal](Vector2 pos)
    {
        return interp::map_range(
            pos, {{}, dims_f64},
            {{-1.0, -aspect_ratio_reciprocal}, {1.0, aspect_ratio_reciprocal}});
    };

    for (auto& particle : ps)
    {
        particle.pos    = rand.vector({.min = Vector2{0.0, 0.0}, .max = dims_f64});
        particle.radius = radius;
    }


    const auto update = [&]
    {
        for (auto [pos, force] : forces.enumerate_2d())
        {
            const auto noisy_angle =
                noise::perlin_2d(pos.cast<f64>() +
                                     Vector2::combine(10) * static_cast<f64>(frame_counter * 0.01),
                                 {.scale = scale, .roughness = 1.0}) *
                math::two_pi;
            // TODO frame_counter is always zero here, how were we using it? make flow field slowly
            // change?
            force = Vector2::from_polar({.length = max_force, .angle = noisy_angle});
        }

        for (auto [i, particle] : ranges::views::enumerate(ps))
        {
            const auto index = particle.pos.cast<u64>();
            auto force       = forces.at_or(index, Vector2{0.0, 0.0});
            //     // TODO what if we zero out prev vel and let it move only due to current force
            //     // particle.vel     = Vector2{};

            //     // if (math::distance(field_to_graph(particle.pos), window.mouse.pos) <=
            //     mouse_dist)
            //     // {
            //     //     particle.vel += rand.polar_vector({1.0, 2.0});
            //     // }
            particle.apply_force(force);
        }

        ps.update(watch.seconds() * sim_speed);
        watch.reset();

        // // wrap position
        // // TODO draw tiled, so the wrapping is obvious and looks beautiful
        ps.for_each(
            [=](Particle<>& particle)
            {
                particle.pos.x = math::wrap_max(particle.pos.x, static_cast<f64>(dims.x));
                particle.pos.y = math::wrap_max(particle.pos.y, static_cast<f64>(dims.y));
                particle.vel.clamp_length({0.0, max_speed});
            });

        frame_counter++;
    };

    const auto draw = [&]
    {
        draw::background("#000000"_c);

        for (const auto& particle : ps)
        {
            // when vel is -ve, it gives a cool shadow effect coz of overflow!!!
            const auto col = (particle.vel.normalized() * 255.0).cast<u8>();
            // const auto col = (particle.vel.abs().normalized() * 255.0).cast<u8>();
            const auto colorr = Color{col.x, 0, col.y};
            // const auto colorr = "#ffffff"_c;

            draw::circle(window, Circle{field_to_graph(particle.pos), particle.radius}, colorr, 3);

            // drawing a circle around the mouse gives a cool following effect for some reason
            // draw::circle(window, Circle{window.mouse.pos, mouse_dist},
            //              {.fill_color = "#8400ff05"_c});
        }

        // window.pan();
        window.zoom_to_cursor();

        file::write(file::pam, window.get_image(),
                    fmt::format("./exports/{:05}.pam", frame_counter));
    };

    auto watch1 = Stopwatch{};
    run(window, update, draw);
    print(frame_counter / watch1.seconds(), "fps");
    // TODO fps seems to scale inversely with particle count

    // TODO add some bloom for the orange particles
}
