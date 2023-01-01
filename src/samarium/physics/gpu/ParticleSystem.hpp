/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <array>  // for to_array
#include <span>   // for span
#include <vector> // for vector

#include "samarium/gl/Shader.hpp"        // for ComputeShader
#include "samarium/gl/Vertex.hpp"        // for ShaderStorageBuffer
#include "samarium/physics/Particle.hpp" // for Particle
#include "samarium/util/Result.hpp"      // for expect

namespace sm::gpu
{
struct ParticleSystem
{
    struct Buffers
    {
        gl::ShaderStorageBuffer particles{};
        gl::ShaderStorageBuffer delta_time{};
    };

    struct Shaders
    {
        gl::ComputeShader update{expect(gl::ComputeShader::make(
#include "version.comp.glsl"

#include "Particle.comp.glsl"

#include "update.comp.glsl"
            ))};
    };

    std::vector<Particle<f32>> particles{};
    Buffers buffers{};
    Shaders shaders{};

    explicit ParticleSystem(u64 size, const Particle<f32>& default_particle = {})
        : particles(size, default_particle)
    {
    }

    void send_to_gpu() { buffers.particles.set_data(std::span(particles), gl::Usage::StaticCopy); }

    void fetch_from_gpu() { buffers.particles.read_to(std::span(particles)); }

    void update(f32 delta_time = 0.01)
    {
        buffers.particles.bind(0);
        send_to_gpu();

        buffers.delta_time.bind(1);
        buffers.delta_time.set_data(std::to_array({delta_time}));

        shaders.update.bind();
        shaders.update.set("delta_time", delta_time);
        shaders.update.run(static_cast<u32>(particles.size()));
        fetch_from_gpu();
    }
};
} // namespace sm::gpu
