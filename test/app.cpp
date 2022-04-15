/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/samarium.hpp"
#include "samarium/graphics/Image.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/geometry.hpp"
#include "samarium/physics/collision.hpp"
#include "samarium/util/Stopwatch.hpp"

using namespace sm;
using namespace sm::literals;

auto print_float(f64 value) { fmt::print("{:8} ", value); }

struct Params
{
    Vector2 gravity{};
    f64 spring_stiffness{};
    f64 spring_damping{};
    f64 particle_mass{};
    f64 particle_radius{};
    Vector2 particle_velocity{};
    Dimensions dims{};
};

int main()
{
    const auto params = Params{.gravity          = -30.0_y,
                               .spring_stiffness = 80.0,
                               .spring_damping   = 15.0,
                               .particle_mass    = 0.6,
                               .particle_radius  = 0.7,
                               .dims             = {2, 2}};

    auto particles = Grid<Dual<Particle>>::generate(
        params.dims,
        [&](auto indices)
        {
            const auto x = interp::map_range<f64>(static_cast<f64>(indices.x),
                                                  Extents<u64>{0UL, params.dims.x}.as<f64>(),
                                                  Extents<f64>{-10, 10});

            const auto y = interp::map_range<f64>(static_cast<f64>(indices.y),
                                                  Extents<u64>{0UL, params.dims.y}.as<f64>(),
                                                  Extents<f64>{-20, 20});

            auto pos = Vector2{x, y};
            pos.rotate(1);

            return Dual<Particle>{{.pos    = pos,
                                   .vel    = params.particle_velocity,
                                   .radius = params.particle_radius,
                                   .mass   = params.particle_mass,
                                   .color  = colors::red}};
        });

    auto springs = [&]
    {
        std::vector<Spring> temp;
        temp.reserve(params.dims.x * params.dims.y * 4);

        for (auto i : range(params.dims.y))
        {
            for (auto j : range(params.dims.x))
            {
                if (j != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i}].now); }
                if (i != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j, i - 1}].now); }
                if (i != 0 && j != 0)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i - 1}].now,
                                      params.spring_stiffness, params.spring_damping);
                }
                if (i != 0 && j != params.dims.x - 1)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j + 1, i - 1}].now,
                                      params.spring_stiffness, params.spring_damping);
                }
            }
        }

        return temp;
    }();

    auto app = App{{.dims = dims720}};

    const auto viewport_box = app.viewport_box();

    auto watch = Stopwatch{};

    const auto update = [&](auto delta)
    {
        for (auto&& spring : springs) { spring.update(); }

        const auto dot = std::abs(
            Vector2::dot(viewport_box[3].vector().normalized().rotated_by(std::numbers::pi / 2.0),
                         particles[2]->vel));

        if (dot < 0.1) print("dot");

        for (auto&& particle : particles)
        {
            particle->apply_force(particle->mass * params.gravity);
            particle->update(delta);
            for (auto&& other_particle : particles)
            {
                phys::collide(particle.now, other_particle.now);
            }

            for (auto&& wall : viewport_box) { phys::collide(particle, wall); }
        }
    };

    const auto draw = [&]
    {
        app.fill("#16161c"_c);
        for (const auto& spring : springs)
        {
            app.draw_line_segment(LineSegment{spring.p1.pos, spring.p2.pos},
                                  colors::white.with_multiplied_alpha(0.8), 0.04);
        }

        for (auto&& particle : particles)
        {
            app.draw(particle.now);
            particle.prev = particle.now;
        }

        fmt::print("Framerate: {}\n", std::round(1.0 / watch.time().count()));
        watch.reset();
    };

    app.run(update, draw, 5);
}
