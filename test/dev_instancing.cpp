/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;

auto main() -> i32
{
    auto window = Window{{.dims = {1280, 720}}};
    auto bench  = Benchmark{};

    auto& ctx = window.context;

    ctx.vert_sources.emplace("PosInstance",
#include "samarium/gl/shaders/PosInstance.vert.glsl"
    );

    ctx.shaders.emplace(
        "PosInstance",
        gl::Shader{expect(gl::VertexShader::make(ctx.vert_sources.at("PosInstance"))),
                   expect(gl::FragmentShader::make(ctx.frag_sources.at("Pos")))});
    const auto& shader = ctx.shaders.at("PosInstance");
    ctx.set_active(shader);


    ctx.vertex_arrays.emplace("PosInstance", gl::VertexArray{{}});

    const auto circle      = Circle{{0, 0}, 0.3};
    const auto color       = Color{100, 60, 255};
    const auto point_count = 64;
    const auto points      = math::regular_polygon_points<f32>(point_count, circle);

    glEnableVertexAttribArray(0);
    GLuint points_vertex_buffer;
    glGenBuffers(1, &points_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, points_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vector2f) * points.size(), &points[0], GL_STATIC_DRAW);


    auto pts = std::vector<Vector2f>{{-0.4F, -0.4F}, {-0.4F, 0.4F}, {0.4F, 0.4F}, {0.4F, -0.4F}};
    auto particle_count = pts.size();
    unsigned int instances_buffer;
    glGenBuffers(1, &instances_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, instances_buffer);
    glBufferData(GL_ARRAY_BUFFER, particle_count * sizeof(Vector2f), &pts[0],
                 GL_STATIC_DRAW /* GL_STREAM_DRAW */);

    // ctx.vertex_arrays.at("PosInstance").bind();
    // vertex attributes
    // glEnableVertexAttribArray(1);
    // glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, /* 2 *  */ sizeof(Vector2f) /* or zero? */,
    //                       (void*)0);
    // glVertexAttribDivisor(1, 1);
    // glBindVertexArray(0);


    const auto draw_circles = [&]
    {
        const auto& shader = ctx.shaders.at("PosInstance");
        ctx.set_active(shader);


        ctx.vertex_arrays.at("PosInstance").bind();

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, points_vertex_buffer);
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
        glBindBuffer(GL_ARRAY_BUFFER, instances_buffer);
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
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, static_cast<i32>(points.size()),
                              static_cast<i32>(particle_count));
        bench.add("gl draw instanced");
    };

    const auto draw = [&]
    {
        draw::background(Color{});

        draw_circles();
        // TODO comment out below: then flickers A LOT, ignores view
        // draw::circle(window, {{0.1, 0.2}, 0.1}, Color{255, 255, 0});


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
