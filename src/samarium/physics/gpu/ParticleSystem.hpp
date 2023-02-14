/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <span> // for span

#include "fmt/format.h" // for to_string

#include "samarium/gl/Shader.hpp"           // for ComputeShader
#include "samarium/gui/Window.hpp"          // for Window
#include "samarium/math/vector_math.hpp"    // for regular_polygon_points
#include "samarium/physics/Particle.hpp"    // for Particle
#include "samarium/util/Result.hpp"         // for expect
#include "samarium/util/replace_substr.hpp" // for replace_substr

namespace sm::gpu
{
struct ParticleSystem
{
    struct Shaders
    {
        static constexpr auto update_src =
#include "Particle.comp.glsl"

#include "update.comp.glsl"
            ;
        gl::ComputeShader update;
    };

    gl::MappedBuffer<Particle<f32>> particles;
    i32 shader_local_size;
    Shaders shaders;

    explicit ParticleSystem(i32 size,
                            const Particle<f32>& default_particle = {},
                            i32 compute_shader_local_size         = 16)
        : particles{size, default_particle}, shader_local_size{compute_shader_local_size},
          shaders{.update{Shaders::update_src, shader_local_size}}
    {
    }

    void update(f32 delta_time = 0.01F)
    {
        const auto work_group_count =
            (static_cast<i32>(particles.data.size()) + shader_local_size - 1) / shader_local_size;
        shaders.update.bind();
        particles.bind(2);
        shaders.update.set("delta_time", delta_time);
        shaders.update.run(static_cast<u32>(work_group_count));
    }

    void draw(Window& window, Color color, f32 scale = 1.0F, u32 point_count = 16)
    {
        const auto points = math::regular_polygon_points<f32>(point_count, {{}, 1.0F});

        const auto& shader = window.context.shaders.at("particles");
        window.context.set_active(shader);
        shader.set("scale", scale);
        shader.set("view", window.view);
        shader.set("color", color);

        const auto& buffer = window.context.vertex_buffers.at("default");

        auto& vao = window.context.vertex_arrays.at("Pos");
        window.context.set_active(vao);
        buffer.set_data(points);
        vao.bind(buffer, sizeof(Vector2_t<f32>));

        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, static_cast<i32>(points.size()),
                              static_cast<i32>(particles.data.size()));
    }
};
} // namespace sm::gpu
