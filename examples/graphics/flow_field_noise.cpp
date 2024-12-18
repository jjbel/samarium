/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "range/v3/view/enumerate.hpp"
#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"


using namespace sm;
using namespace sm::literals;

constexpr auto window_dims     = Dimensions{1000, 1000};
constexpr u64 downscale_factor = 20;
constexpr u64 particle_count   = 10'000;
constexpr f64 radius           = 0.007;
constexpr f64 transparency     = 1.0;
constexpr f64 sim_speed        = 0.5;
constexpr f64 max_speed        = 4;
constexpr f64 max_force        = 1.0;
constexpr f64 mouse_dist       = 0.05;

// TODO fps seems to scale inversely with particle count
// TODO add some bloom for the orange particles
// TODO also see old values (the ones in 1920x1080). gave better results

// TODO make f64 a typedef and compare performance throughout
// TODO make f32 <-> f64 vector conversion functions

auto main() -> i32
{
    auto window        = Window{{.dims = window_dims}};
    auto watch         = Stopwatch{};
    auto frame_counter = 0;

    const auto scale = .4;
    const auto dims  = window.dims / downscale_factor; // make this window dims / detail

    auto ps     = ParticleSystem{particle_count};
    auto forces = VectorField{dims};
    auto rand   = RandomGenerator{1'000'000, RandomMode::Stable, 42};

    const auto dims_f64 = dims.template cast<f64>();

    // instead of doing any mapping, should change the window view
    // TODO VVIMP mapping pixel space to graph space.
    const auto aspect_ratio_reciprocal = window.dims.template cast<f64>().slope();
    const auto field_to_graph          = [dims_f64, aspect_ratio_reciprocal](Vec2 pos)
    {
        return interp::map_range(
            pos, {{}, dims_f64},
            {{-1.0, -aspect_ratio_reciprocal}, {1.0, aspect_ratio_reciprocal}});
    };

    for (auto& particle : ps)
    {
        particle.pos    = rand.vector({.min = Vec2{0.0, 0.0}, .max = dims_f64});
        particle.radius = radius;
    }


    auto bench = Benchmark{};

    const auto update = [&](f64 _)
    {
        bench.reset();
        for (auto [pos, force] : forces.enumerate_2d())
        {
            const auto noisy_angle =
                noise::perlin_2d(pos.template cast<f64>() +
                                     Vec2::combine(10) * static_cast<f64>(frame_counter * 0.01),
                                 {.scale = scale, .roughness = 1.0}) *
                math::two_pi;
            // TODO frame_counter is always zero here, how were we using it? make flow field slowly
            // change?
            force = Vec2::from_polar({.length = max_force, .angle = noisy_angle});
        }

        bench.add("update forces");

        // TODO what if we just rotate the forces at the same speed
        // TODO what if we make it a velocity field instead of a force field

        for (auto [i, particle] : ranges::views::enumerate(ps))
        {
            const auto index = particle.pos.template cast<u64>();
            auto force       = forces.at_or(index, Vec2{0.0, 0.0});
            //     // TODO what if we zero out prev vel and let it move only due to current force
            //     // particle.vel     = Vec2{};

            //     // if (math::distance(field_to_graph(particle.pos), window.mouse.pos) <=
            //     mouse_dist)
            //     // {
            //     //     particle.vel += rand.polar_vector({1.0, 2.0});
            //     // }
            particle.apply_force(force);
        }

        bench.add("apply forces");

        ps.update(watch.seconds() * sim_speed);
        watch.reset();

        bench.add("particles update");


        // // wrap position
        // // TODO draw tiled, so the wrapping is obvious and looks beautiful
        ps.for_each(
            [=](Particle<>& particle)
            {
                particle.pos.x = math::wrap_max(particle.pos.x, static_cast<f64>(dims.x));
                particle.pos.y = math::wrap_max(particle.pos.y, static_cast<f64>(dims.y));
                particle.vel.clamp_length({0.0, max_speed});
            });

        bench.add("particles wrap");

        frame_counter++;
    };

    auto image = Image{window.dims};

    const auto draw = [&]
    {
        draw::background("#000000"_c);
        bench.add("draw background");

        for (const auto& particle : ps)
        {
            // when vel is -ve, it gives a cool shadow effect coz of overflow!!!
            const auto col = (particle.vel.normalized() * 255.0).template cast<u8>();
            // const auto col = (particle.vel.abs().normalized() * 255.0).template cast<u8>();
            const auto colorr = Color{col.x, 0, col.y};
            // const auto colorr = "#ffffff"_c;

            draw::circle(window, Circle{field_to_graph(particle.pos), particle.radius}, colorr, 3);

            // drawing a circle around the mouse gives a cool following effect for some reason
            // draw::circle(window, Circle{window.mouse.pos, mouse_dist},
            //              {.fill_color = "#8400ff05"_c});
        }

        bench.add("draw particles");


        // window.pan();
        window.zoom_to_cursor();
        bench.add("zoom to cursor");

        // file::write(file::pam, window.get_image(),
        //             fmt::format("./exports/{:05}.pam", frame_counter));
        // bench.add("pam export");

        // window.get_image(image);
        // file::write(file::pam, image, fmt::format("./exports/{:05}.pam", frame_counter));
        // bench.add("pam export to target");

        bench.add_frame();
    };

    run(window, update, draw);
    bench.print();
}
