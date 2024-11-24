/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <span>
#include <vector>

#include "samarium/core/types.hpp"
#include "samarium/gui/Window.hpp"
#include "samarium/math/Vec2.hpp"
#include "samarium/util/Benchmark.hpp"

#include "Vertex.hpp"
#include "gl.hpp"

namespace sm::gl
{
struct Instancer
{
    Window& window;

    std::vector<Vec2f> geometry{};
    std::vector<Vec2f> instances_pos{};
    u32 geometry_vb{};
    u32 instances_vb{};

    Benchmark bench{};

    /*
    conclusion:
    for 64 verts:
    window.display() and draw instance take:
    100K particles: 5.1ms, 0.3ms
        1M particles: 28-55ms, 2ms
        10M particles: 440ms, 45ms

    for 3 verts:
    10M particles: 19ms, 19ms

    avoid copying to instancer.instances_pos
    (hence use data oriented design: store pos in instancer.instances_pos only)
    copying from instancer.instances_pos to buffer is unavoidable
    no need to copy to/from instancer.geometry

    TODO: use gl classes
    TODO: have to set vertattrib every call?
    */

    Instancer(Window& window, std::span<const Vec2f> geometry, std::span<const Vec2f> instances_pos)
        : window{window}, geometry(geometry.begin(), geometry.end()),
          instances_pos(instances_pos.begin(), instances_pos.end())
    {
        const auto& shader = window.context.shaders.at("PosInstance");
        window.context.set_active(shader);

        window.context.vertex_arrays.emplace("PosInstance", gl::VertexArray{{}});

        glEnableVertexAttribArray(0);
        glGenBuffers(1, &geometry_vb);
        glBindBuffer(GL_ARRAY_BUFFER, geometry_vb);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vec2f) * geometry.size(), &geometry[0],
                     GL_STATIC_DRAW);

        glGenBuffers(1, &instances_vb);
        glBindBuffer(GL_ARRAY_BUFFER, instances_vb);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vec2f) * instances_pos.size(), &instances_pos[0],
                     GL_STATIC_DRAW /* GL_STREAM_DRAW */);

        // ctx.vertex_arrays.at("PosInstance").bind();
        // vertex attributes
        // glEnableVertexAttribArray(1);
        // glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, /* 2 *  */ sizeof(Vec2f) /* or zero?
        // */,
        //                       (void*)0);
        // glVertexAttribDivisor(1, 1);
        // glBindVertexArray(0);
    }

    void draw(Color color)
    {
        bench.reset();
        glBindBuffer(GL_ARRAY_BUFFER, instances_vb);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vec2f) * instances_pos.size(), &instances_pos[0],
                     GL_STATIC_DRAW /* GL_STREAM_DRAW */);
        bench.add("update positions buffer");

        const auto& shader = window.context.shaders.at("PosInstance");
        window.context.set_active(shader);

        window.context.vertex_arrays.at("PosInstance").bind();

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, geometry_vb);
        glVertexAttribPointer(
            0, // attribute. No particular reason for 0, but must match the layout in the shader.
            2, // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,        // stride
            (void*)0  // array buffer offset
        );

        // 2nd attribute buffer : positions of particles' centers
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, instances_vb);
        glVertexAttribPointer(
            1, // attribute. No particular reason for 1, but must match the layout in the shader.
            2, // size : x, y => 2
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,        // stride
            (void*)0  // array buffer offset
        );
        glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
        glVertexAttribDivisor(1, 1); // positions : one per quad (its center) -> 1

        // TODO need to do every frame?
        shader.set("color", color);
        shader.set_transform("view", window.world2gl());
        bench.add("gl uniforms");

        bench.add("gl state update");
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, static_cast<i32>(geometry.size()),
                              static_cast<i32>(instances_pos.size()));
        bench.add("gl draw instanced");
    }
};
} // namespace sm::gl
