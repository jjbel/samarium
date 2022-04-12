/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/samarium.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/interp.hpp"
#include "samarium/physics/Particle.hpp"
#include "samarium/util/random.hpp"

using namespace sm;
using namespace sm::literals;

struct Spring
{
    Particle& p1;
    Particle& p2;
    const f64 rest_length;
    f64 spring_constant;
    f64 damping_constant;

    Spring(Particle& particle1,
           Particle& particle2,
           f64 spring_constant_  = 100,
           f64 damping_constant_ = 1900)
        : p1{particle1}, p2{particle2}, rest_length{math::distance(particle1.pos, particle2.pos)},
          spring_constant{spring_constant_}, damping_constant{damping_constant_}
    {
    }

    [[nodiscard]] auto length() const { return math::distance(p1.pos, p2.pos); }

    auto update()
    {
        const auto dx = this->length() - rest_length;

        const auto norm = p2.pos - p1.pos;

        const auto spring = interp::clamp(spring_constant * dx, {-10000, 10000});
        const auto damping =
            interp::clamp(Vector2::dot(p2.vel - p1.vel, norm) * damping_constant, {-10000, 10000});
        print(spring, damping);

        const auto force = norm.with_length(spring + damping);
        p1.apply_force(force);
        p2.apply_force(-force);
    }
};

int main()
{
    const auto gravity = -10.0_y;

    const auto dims = Dimensions{4,5};

    auto particles = Grid<Dual<Particle>>::generate(
        dims,
        [&](Indices indices)
        {
            const auto x = interp::map_range<f64>(indices.x, Extents<u64>{0UL, dims.x}.as<f64>(),
                                                  Extents<f64>{-5, 5});

            const auto y = interp::map_range<f64>(indices.y, Extents<u64>{0UL, dims.y}.as<f64>(),
                                                  Extents<f64>{-5, 5});

            auto pos = Vector2{x, y};
            pos.rotate(1);

            return Dual<Particle>{{.pos    = pos,
                                   .vel    = Vector2{0, -20},
                                   .radius = .3,
                                   .mass   = 46,
                                   .color  = colors::red}};
        });

    auto springs = [&]
    {
        std::vector<Spring> temp;
        temp.reserve(dims.x * dims.y * 4);

        for (auto i : range(dims.y))
        {
            for (auto j : range(dims.x))
            {
                if (j != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i}].now); }
                if (i != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j, i - 1}].now); }
                if (i != 0 && j != 0)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i - 1}].now);
                }
                if (i != 0 && j != dims.x - 1)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j + 1, i - 1}].now);
                }
            }
        }

        return temp;
    }();

    print(springs.size());

    auto app = App{{.dims = dims720}};

    const auto viewport_box = app.viewport_box();

    const auto update = [&](auto delta)
    {
        app.fill("#16161c"_c);

        for (auto&& spring : springs) { spring.update(); }

        for (auto&& particle : particles)
        {
            particle->apply_force(particle->mass * gravity);

            particle->update(delta);

            for (auto&& wall : viewport_box) { phys::collide(particle, wall); }
        }
    };

    const auto draw = [&]
    {
        for (const auto& spring : springs)
        {
            app.draw_line_segment(LineSegment{spring.p1.pos, spring.p2.pos}, colors::white, 0.02);
        }

        for (auto&& particle : particles)
        {
            app.draw(particle.now);
            particle.prev = particle.now;
        }

        fmt::print("\n{}: ", app.frame_counter);
    };

    const auto combined = [&]
    {
        update(1.0);
        draw();
    };

    app.run(update, draw, 1);
}
