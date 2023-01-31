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
#include "samarium/gl/Sync.hpp"          // for Sync
#include "samarium/gl/Vertex.hpp"        // for ShaderStorageBuffer
#include "samarium/physics/Particle.hpp" // for Particle
#include "samarium/util/Result.hpp"      // for expect
#include "samarium/util/Stopwatch.hpp"

namespace sm::gpu
{
struct ParticleSystem
{
    struct Shaders
    {
        gl::ComputeShader update{expect(gl::ComputeShader::make(
#include "version.comp.glsl"

#include "Particle.comp.glsl"

#include "update.comp.glsl"
            ))};
    };

    gl::MappedBuffer<Particle<f32>> particles;
    Shaders shaders{};

    explicit ParticleSystem(u64 size, const Particle<f32>& default_particle = {})
        : particles{expect(
              gl::MappedBuffer<Particle<f32>>::make(static_cast<i32>(size), default_particle))}
    {
    }

    void update(f32 delta_time = 0.01F)
    {
        const auto work_group_count = (particles.data.size() + 64 - 1) / 64;
        shaders.update.bind();
        particles.bind(2);
        shaders.update.set("delta_time", delta_time);
        shaders.update.run(static_cast<u32>(work_group_count));
    }
};
} // namespace sm::gpu
