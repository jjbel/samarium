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

using namespace sm;
using namespace sm::literals;

auto print_float(f64 value) { fmt::print("{:8} ", value); }

struct Spring
{
    Particle& p1;
    Particle& p2;
    const f64 rest_length;
    const f64 stiffness;
    const f64 damping;

    Spring(Particle& particle1,
           Particle& particle2,
           f64 stiffness_ = 100.0,
           f64 damping_   = 10.0) noexcept
        : p1{particle1}, p2{particle2}, rest_length{math::distance(particle1.pos, particle2.pos)},
          stiffness{stiffness_}, damping{damping_}
    {
    }

    [[nodiscard]] auto length() const noexcept { return math::distance(p1.pos, p2.pos); }

    auto update() noexcept
    {
        const auto vec    = p2.pos - p1.pos;
        const auto spring = (vec.length() - rest_length) * stiffness;
        auto damp         = Vector2::dot(vec.normalized(), p2.vel - p1.vel) * damping;

        if (std::abs(damp) < 0.01) { damp = 0.0; }
        else
            print("Uh-oh");

        const auto force = vec.with_length(spring + damp);

        fmt::print(R"(
Vec:    {},
Length: {:5},
Spring: {:5},
Damp:   {:5},
Force:  {}
)",
                   vec, vec.length(), spring, damp, force);
        // print(p1.vel, p2.vel, p2.vel - p1.vel);

        p1.apply_force(force);
        p2.apply_force(-force);
    }
};

struct Params
{
    Vector2 gravity{};
    f64 spring_stiffness{};
    f64 spring_damping{};
    f64 particle_mass{};
    f64 particle_radius{};
    Dimensions dims{};
};

int main()
{
    const auto params = Params{.gravity          = -30.0_y,
                               .spring_stiffness = 10.0,
                               .spring_damping   = 20.0,
                               .particle_mass    = 1.0,
                               .particle_radius  = 0.9,
                               .dims             = {4, 4}};

    auto particles = Grid<Dual<Particle>>::generate(
        params.dims,
        [&](auto indices)
        {
            const auto x = interp::map_range<f64>(static_cast<f64>(indices.x),
                                                  Extents<u64>{0UL, params.dims.x}.as<f64>(),
                                                  Extents<f64>{-10, 10});

            const auto y = interp::map_range<f64>(static_cast<f64>(indices.y),
                                                  Extents<u64>{0UL, params.dims.y}.as<f64>(),
                                                  Extents<f64>{-10, 10});

            auto pos = Vector2{x, y};
            pos.rotate(1);

            return Dual<Particle>{{.pos    = pos,
                                   .vel    = Vector2{10, 20},
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

    const auto update = [&](auto delta)
    {
        app.fill("#16161c"_c);

        fmt::print("\n{}:\n", app.frame_counter);
        for (auto&& spring : springs) { spring.update(); }

        for (auto&& particle : particles)
        {
            particle->apply_force(particle->mass * params.gravity);
            // particle->vel.clamp_length({0.0, 40.0});
            particle->update(delta);
            for (auto&& particle_ : particles)
            {
                if (&particle != &particle_) { phys::collide(particle.now, particle_.now); }
            }

            for (auto&& wall : viewport_box) { phys::collide(particle, wall); }
        }
    };

    const auto draw = [&]
    {
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
    };

    app.run(update, draw, 1);
}
