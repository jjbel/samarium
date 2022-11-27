/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <concepts>
#include <memory> // for allocator_trai...
#include <span>   // for span
#include <vector> // for vector

#include "BS_thread_pool.hpp"                         // for multi_future
#include "range/v3/algorithm/for_each.hpp"            // for for_each
#include "range/v3/algorithm/sort.hpp"                // for sort
#include "range/v3/functional/identity.hpp"           // for identity
#include "range/v3/iterator/basic_iterator.hpp"       // for basic_iterator
#include "range/v3/iterator/unreachable_sentinel.hpp" // for operator==
#include "range/v3/view/enumerate.hpp"                // for enumerate, enu...
#include "range/v3/view/view.hpp"                     // for view_closure
#include "range/v3/view/zip.hpp"                      // for zip_view
#include "range/v3/view/zip_with.hpp"                 // for iter_zip_with_...
#include "tl/function_ref.hpp"                        // for function_ref

#include "samarium/core/types.hpp"       // for f64, u64, usize
#include "samarium/math/Extents.hpp"     // for Extents, range
#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/physics/Particle.hpp" // for Particle
#include "samarium/util/HashGrid.hpp"    // for HashGrid
#include "samarium/util/ThreadPool.hpp"  // for ThreadPool
#include "samarium/util/util.hpp"        // for project_view

#include "collision.hpp" // for collide

namespace sm
{
template <typename Particle_t = Particle<f64>, u64 CellCapacity = 32> struct ParticleSystem
{
    std::vector<Particle_t> particles;
    HashGrid<u32, CellCapacity> hash_grid;

    /**
     * @brief               Create `size` particles
     *
     * @param  size
     * @param  default_particle
     */
    explicit ParticleSystem(u64 size                           = 100UL,
                            const Particle_t& default_particle = {},
                            f64 cell_size                      = 0.5)
        : particles(size, default_particle), hash_grid{cell_size}
    {
    }

    static auto generate(u64 size, const auto& callable)
    {
        auto output = ParticleSystem(size);
        // https://stackoverflow.com/a/70808634/17100530
        for (auto [i, particle] : ranges::views::enumerate(output.particles))
        {
            particle = callable(i);
        }
        return output;
    }

    void update(f64 time_delta = 1.0) noexcept
    {
        ranges::for_each(particles,
                         [time_delta](Particle_t& particle) { particle.update(time_delta); });
    }

    void update(ThreadPool& thread_pool, f64 time_delta = 1.0) noexcept
    {
        const auto job = [&](auto min, auto max)
        {
            for (auto i : range(min, max)) { particles[i].update(time_delta); }
        };

        thread_pool.parallelize_loop(0UL, particles.size(), job, thread_pool.get_thread_count())
            .wait();
    }

    void apply_force(Vector2 force) noexcept
    {
        ranges::for_each(particles, [force](Particle_t& particle) { particle.apply_force(force); });
    }

    void apply_forces(std::span<Vector2> forces) noexcept
    {
        for (auto i : range(particles.size())) { particles[i].apply_force(forces[i]); }
    }

    void for_each(const auto& callable) { ranges::for_each(particles, callable); }

    /**
     * @brief               Collide the particles with themselves
     *
     * @tparam CellCapacity Max particles in one cell of the hash grid
     * @param  damping      Coefficient of restitution
     * @param  cell_size    Size of cell of hash grid
     * @return Dimensions   [collisions, pairs checked]
     */
    [[maybe_unused]] auto self_collision(f64 damping = 1.0)
    {
        hash_grid.map.clear();
        hash_grid.map.reserve(particles.size());
        for (auto i : range(particles.size())) { hash_grid.insert(particles[i].pos, i); }

        auto count1 = u32{};
        auto count2 = u32{};
        for (auto i : range(particles.size()))
        {
            // Slow: for (auto j : range(particles.size()))
            // TODO bottleneck is still here, not in the actual collision
            for (auto j : hash_grid.neighbors(particles[i].pos))
            {
                // we're looping through ordered pairs, so avoid colliding each pair twice
                if (i < j)
                {
                    count1 += phys::collide(particles[i], particles[j], damping);
                    count2++;
                }
            }
        }
        return Dimensions::make(count1, count2);
    }

    [[nodiscard]] auto operator[](u64 index) noexcept { return particles[index]; }
    [[nodiscard]] auto operator[](u64 index) const noexcept { return particles[index]; }

    [[nodiscard]] auto begin() noexcept { return particles.begin(); }
    [[nodiscard]] auto end() noexcept { return particles.end(); }

    [[nodiscard]] auto begin() const noexcept { return particles.cbegin(); }
    [[nodiscard]] auto end() const noexcept { return particles.cend(); }

    [[nodiscard]] auto cbegin() const noexcept { return particles.cbegin(); }
    [[nodiscard]] auto cend() const noexcept { return particles.cend(); }

    [[nodiscard]] auto size() const noexcept { return particles.size(); }
    [[nodiscard]] auto empty() const noexcept { return particles.empty(); }
};
} // namespace sm
