/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;

struct Instancer
{
    Window& window;
    std::vector<Vector2f> geometry{};
    std::vector<Vector2f> instances_pos{};

    u32 geometry_vb{};
    u32 instances_vb{};

    Instancer(Window& window,
              std::span<const Vector2f> geometry,
              std::span<const Vector2f> instances_pos)
        : window{window}, geometry(geometry.begin(), geometry.end()),
          instances_pos(instances_pos.begin(), instances_pos.end())
    {
        const auto& shader = window.context.shaders.at("PosInstance");
        window.context.set_active(shader);

        window.context.vertex_arrays.emplace("PosInstance", gl::VertexArray{{}});

        glEnableVertexAttribArray(0);
        glGenBuffers(1, &geometry_vb);
        glBindBuffer(GL_ARRAY_BUFFER, geometry_vb);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vector2f) * geometry.size(), &geometry[0],
                     GL_STATIC_DRAW);

        glGenBuffers(1, &instances_vb);
        glBindBuffer(GL_ARRAY_BUFFER, instances_vb);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Vector2f) * instances_pos.size(), &instances_pos[0],
                     GL_STATIC_DRAW /* GL_STREAM_DRAW */);

        // ctx.vertex_arrays.at("PosInstance").bind();
        // vertex attributes
        // glEnableVertexAttribArray(1);
        // glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, /* 2 *  */ sizeof(Vector2f) /* or zero?
        // */,
        //                       (void*)0);
        // glVertexAttribDivisor(1, 1);
        // glBindVertexArray(0);
    }

    void draw(Color color, Benchmark& bench)
    {
        bench.reset();
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

        shader.set("color", color);
        shader.set_transform("view", window.world2gl());
        bench.add("gl uniforms");

        bench.add("gl state update");
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, static_cast<i32>(geometry.size()),
                              static_cast<i32>(instances_pos.size()));
        bench.add("gl draw instanced");
    }
};

auto main() -> i32
{
    auto window = Window{{.dims = {1280, 720}}};
    auto bench  = Benchmark{};

    const auto circle      = Circle{{0, 0}, 0.3};
    const auto color       = Color{100, 60, 255};
    const auto point_count = 64;
    const auto points      = math::regular_polygon_points<f32>(point_count, circle);

    auto pts = std::vector<Vector2f>{{-0.4F, -0.4F}, {-0.4F, 0.4F}, {0.4F, 0.4F}, {0.4F, -0.4F}};

    auto instancer = Instancer(window, points, pts);

    window.camera.scale /= 5;
    const auto draw = [&]
    {
        draw::background(Color{});

        instancer.draw(color, bench);
        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{255, 255, 0});


        // for (const auto& particle : ps)
        // {
        //     const auto colorr = Color{col.x, 0, col.y};
        //     draw::circle(window, Circle{field_to_graph(particle.pos), particle.radius}, colorr,
        //     3);
        // }
        // bench.add("draw particles");

        window.pan();
        window.zoom_to_cursor();
        bench.add("zoom to cursor");
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        bench.add("sleep");
        bench.add_frame();
    };

    run(window, draw);
    bench.print();
}
