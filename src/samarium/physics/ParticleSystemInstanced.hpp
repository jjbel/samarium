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
#include "samarium/util/Stopwatch.hpp"


namespace sm
{
struct ParticleSystemInstanced
{
    // Mike Acton data oriented design https://youtu.be/rX0ItVEVjHc?t=2530
    gl::Instancer instancer;
    std::vector<Vector2f>& pos;
    std::vector<Vector2f> vel;
    std::vector<Vector2f> acc;
    Color color;
    Stopwatch watch{};

    ParticleSystemInstanced(Window& window,
                            u64 count,
                            f32 radius            = 0.01F,
                            Color color           = Color{255, 255, 255},
                            u32 circle_resolution = 16)
        : instancer(window, math::regular_polygon_points<f32>(circle_resolution, {{}, radius}), {}),
          pos(instancer.instances_pos), color(color)
    {
        pos.resize(count);
        vel.resize(count);
        acc.resize(count);
    }

    auto size() const { return pos.size(); }

    void update()
    {
        // TODO when playing back/simulating properly, shd use a const delta
        const auto time_delta = static_cast<f32>(watch.seconds());
        watch.reset();

        for (auto i : loop::end(size()))
        {
            const auto half_dv = acc[i] * time_delta;
            vel[i] += half_dv;
            pos[i] += vel[i] * time_delta;
            vel[i] += half_dv;
            acc[i] = Vector2f{}; // reset acceleration}
        }
    }

    void draw() { instancer.draw(color); }
};
} // namespace sm
