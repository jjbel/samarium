/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>   // for span
#include <vector> // for vector

#include "range/v3/view/enumerate.hpp" // for enumerate, enumerate_fn

#include "samarium/core/types.hpp"       // for u64, f64
#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/util/FunctionRef.hpp" // for Vector2

#include "Particle.hpp" // for Particle

namespace sm
{
class ThreadPool;

struct ParticleSystem
{
    std::vector<Particle> particles;

    /**
     * @brief               Create `size` particles
     *
     * @param  size
     * @param  default_particle
     */
    ParticleSystem(u64 size = 100UL, const Particle& default_particle = {})
        : particles(size, default_particle)
    {
    }

    static auto generate(u64 size, FunctionRef<Particle(u64)> function) -> ParticleSystem
    {
        auto output = ParticleSystem(size);
        for (auto [i, particle] : ranges::views::enumerate(output.particles))
        {
            particle = function(i);
        }
        return output;
        // https://stackoverflow.com/a/70808634/17100530
    }

    void update(f64 time_delta = 1.0) noexcept;
    void update(ThreadPool& thread_pool, f64 time_delta = 1.0) noexcept;

    void apply_force(Vector2 force) noexcept;
    void apply_forces(std::span<Vector2> forces) noexcept;

    void for_each(FunctionRef<void(Particle&)> function);

    void self_collision(f64 damping = 1.0) noexcept;

    auto operator[](u64 index) noexcept { return particles[index]; }
    auto operator[](u64 index) const noexcept { return particles[index]; }

    auto begin() noexcept { return particles.begin(); }
    auto end() noexcept { return particles.end(); }

    auto begin() const noexcept { return particles.cbegin(); }
    auto end() const noexcept { return particles.cend(); }

    auto cbegin() const noexcept { return particles.cbegin(); }
    auto cend() const noexcept { return particles.cend(); }

    auto size() const noexcept { return particles.size(); }
    auto empty() const noexcept { return particles.empty(); }
};
} // namespace sm
