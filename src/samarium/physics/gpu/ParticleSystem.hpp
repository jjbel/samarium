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
        gl::MappedBuffer<Particle<f32>> particles;
        gl::MappedBuffer<f32> delta_time;
    };

    struct Shaders
    {
        gl::ComputeShader update{expect(gl::ComputeShader::make(
#include "version.comp.glsl"

#include "Particle.comp.glsl"

#include "update.comp.glsl"
            ))};
    };

    Buffers buffers;
    Shaders shaders{};

    explicit ParticleSystem(u64 size,
                            const Particle<f32>& default_particle = {},
                            f32 delta_time                        = 0.01)
        : buffers{expect(gl::MappedBuffer<Particle<f32>>::make(static_cast<i32>(size),
                                                               default_particle)),
                  expect(gl::MappedBuffer<f32>::make(1))}
    {
        // buffers.delta_time.bind(1);
        // auto delta_time_data    = std::to_array({delta_time});
        // buffers.delta_time.data = std::span(delta_time_data);
    }

    void update()
    {
        buffers.particles.bind();

        print("updating", buffers.particles.data.size());
        shaders.update.bind();
        shaders.update.run(buffers.particles.data.size());
        print("updated");
    }

    auto particles() { return buffers.particles.data; }
};
} // namespace sm::gpu
