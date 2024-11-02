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


namespace sm
{
struct ParticleSystemInstanced
{
    gl::Instancer instancer;

    std::vector<Vector2f>& pos;
    std::vector<Vector2f> vel;
    std::vector<Vector2f> acc;

    Color color;

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

    void draw() { instancer.draw(color); }

    void update(f32 time_delta = 1.0)
    {
        for (auto i : loop::end(size()))
        {
            const auto half_dv = acc[i] * time_delta;
            vel[i] += half_dv;
            pos[i] += vel[i] * time_delta;
            vel[i] += half_dv;
            acc[i] = Vector2f{}; // reset acceleration}
        }
    }
};
} // namespace sm