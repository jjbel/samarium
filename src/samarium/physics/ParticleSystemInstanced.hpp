/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/gl/Instancer.hpp"
#include "samarium/graphics/Color.hpp"
#include "samarium/gui/Window.hpp"
#include "samarium/util/HashGrid.hpp" // for HashGrid
#include "samarium/util/Stopwatch.hpp"


namespace sm
{
template <u64 CellCapacity = 32> struct ParticleSystemInstanced
{
    // Mike Acton data oriented design https://youtu.be/rX0ItVEVjHc?t=2530
    gl::Instancer instancer;

    std::vector<Vector2f>& pos;
    std::vector<Vector2f> vel;
    std::vector<Vector2f> acc;

    HashGrid<u32, CellCapacity> hash_grid;

    Color color;
    Stopwatch watch{};

    ParticleSystemInstanced(Window& window,
                            u64 count,
                            f64 cell_size         = 0.1,
                            f32 radius            = 0.01F,
                            Color color           = Color{255, 255, 255},
                            u32 circle_resolution = 16)
        : instancer(window, math::regular_polygon_points<f32>(circle_resolution, {{}, radius}), {}),
          pos(instancer.instances_pos), hash_grid{cell_size}, color(color)
    {
        pos.resize(count);
        vel.resize(count);
        acc.resize(count);
    }

    auto size() const { return pos.size(); }

    auto self_collision(f64 damping = 1.0)
    {
        hash_grid.map.clear();
        hash_grid.map.reserve(size());
        for (auto i : loop::end(size())) { hash_grid.insert(pos[i].cast<f64>(), u32(i)); }

        auto count1 = u32{};
        auto count2 = u32{};
        for (auto i : loop::end(size()))
        {
            // Slow: for (auto j : loop::end(particles.size()))
            // TODO bottleneck is still here, not in the actual collision
            for (auto j : hash_grid.neighbors(pos[i].cast<f64>()))
            {
                // we're looping through ordered pairs, so avoid colliding each pair twice
                if (i < j)
                {
                    // count1 += phys::collide(particles[i], particles[j], damping);
                    count2++;

                    // const auto c = static_cast<u8>(std::abs(g) * 100);
                    const auto c = static_cast<u8>(255);
                    draw::line_segment(instancer.window, {pos[i].cast<f64>(), pos[j].cast<f64>()},
                                       Color{c, c, c}, 0.005F);
                }
            }
        }
        print(count2, "/", size() * size() / 2);
        // return Dimensions::make(count1, count2);
    }

    void update(f32 time_delta)
    {
        // when playing back/simulating properly, shd use a const delta
        for (auto i : loop::end(size()))
        {
            const auto half_dv = acc[i] * time_delta;
            vel[i] += half_dv;
            pos[i] += vel[i] * time_delta;
            vel[i] += half_dv;
            acc[i] = Vector2f{}; // reset acceleration}
        }
    }

    void update()
    {
        const auto time_delta = static_cast<f32>(watch.seconds());
        watch.reset();
        update(time_delta);
    }

    void draw() { instancer.draw(color); }
};
} // namespace sm
