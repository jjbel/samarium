/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>   // for span
#include <vector> // for vector

#include "range/v3/algorithm/fill.hpp" // for fill, fill_fn
#include "range/v3/view/enumerate.hpp" // for enumerate, enumerate_fn

#include "samarium/core/types.hpp"       // for u64, f64
#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/util/FunctionRef.hpp" // for Vector2
#include "samarium/util/HashGrid.hpp"    // for HashGrid
#include "samarium/util/ThreadPool.hpp"  // for ThreadPool
#include "samarium/util/print.hpp"       // for FunctionRef

#include "Particle.hpp"  // for Particle
#include "collision.hpp" // for collide

namespace sm
{
struct ParticleSystem
{
    std::vector<Particle> particles;

    /**
     * @brief               Create `size` particles
     *
     * @param  size
     * @param  default_particle
     */
    explicit ParticleSystem(u64 size = 100UL, const Particle& default_particle = {})
        : particles(size, default_particle)
    {
    }

    static auto generate(u64 size, FunctionRef<Particle(u64)> function)
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

    template <usize MaxParticlesInCell = 32>
    void self_collision(f64 damping = 1.0, f64 cell_size = 1.0) noexcept
    {
        auto hash_grid = HashGrid<u64, MaxParticlesInCell>{cell_size};
        hash_grid.map.reserve(particles.size());
        for (auto i : range(particles.size())) { hash_grid.insert(particles[i].pos, i); }

        usize count = 0;
        for (auto i : range(particles.size()))
        {
            for (auto j : hash_grid.neighbors(particles[i].pos))
            {
                if (i != j)
                {
                    phys::collide(particles[i], particles[j], damping);
                    count++;
                }
            }
        }
        print("# collisions:", count);
        print(hash_grid.map.size());
    }

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

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_PARTICLE_SYSTEM_IMPL)

#include "range/v3/algorithm/for_each.hpp" // for for_each
#include "range/v3/algorithm/sort.hpp"     // for sort

#include "samarium/math/Extents.hpp"     // for Extents, Extents<>::Iter...
#include "samarium/physics/Particle.hpp" // for Particle
#include "samarium/util/FunctionRef.hpp" // for FunctionRef
#include "samarium/util/util.hpp"        // for project_view

namespace sm
{
void ParticleSystem::update(f64 time_delta) noexcept
{
    ranges::for_each(particles, [time_delta](Particle& particle) { particle.update(time_delta); });
}

void ParticleSystem::update(ThreadPool& thread_pool, f64 time_delta) noexcept
{
    const auto job = [&](auto min, auto max)
    {
        for (auto i : range(min, max)) { particles[i].update(time_delta); }
    };

    thread_pool.parallelize_loop(0UL, particles.size(), job, thread_pool.get_thread_count()).wait();
}

void ParticleSystem::apply_force(Vector2 force) noexcept
{
    ranges::for_each(particles, [force](Particle& particle) { particle.apply_force(force); });
}

void ParticleSystem::apply_forces(std::span<Vector2> forces) noexcept
{
    for (auto i : range(particles.size())) { particles[i].apply_force(forces[i]); }
}

void ParticleSystem::for_each(FunctionRef<void(Particle&)> function)
{
    ranges::for_each(particles, function);
}
}; // namespace sm
#endif
