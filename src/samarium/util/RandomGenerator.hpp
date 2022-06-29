/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <initializer_list> // for initializer_list
#include <random>           // for random_device
#include <vector>           // for vector

#include "range/v3/algorithm/generate.hpp" // for generate, generate_fn

#include "samarium/core/types.hpp"       // for f64, u64
#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/Extents.hpp"     // for Extents
#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/math/math.hpp"        // for pi

namespace sm
{
enum class RandomMode
{
    Stable,
    NonDeterministic
};

// PCG-based random number generator, see https://www.pcg-random.org/
struct RandomGenerator
{
    static constexpr auto magic_number = 6364136223846793005ULL;

    std::vector<f64> cache;
    u64 state;
    u64 inc;
    u64 current_index{};

    explicit RandomGenerator(u64 cache_size  = 1024UL,
                             RandomMode mode = RandomMode::Stable,
                             u64 new_inc     = 69) noexcept
        : cache(cache_size), inc{new_inc}
    {
        auto new_state = 69UL;
        if (mode == RandomMode::NonDeterministic)
        {
            new_state = static_cast<u64>(std::random_device{}()); // true randomness
        }
        state = new_state * magic_number + (new_inc | 1);

        ranges::generate(cache, [this] { return this->next_scaled(); });
    }

    void resize(u64 new_size);

    void reseed(u64 new_seed);

    [[nodiscard]] auto next() noexcept -> u64;

    [[nodiscard]] auto next_scaled() noexcept -> f64;

    [[nodiscard]] auto random() -> f64;
    [[nodiscard]] auto operator()() -> f64;

    template <typename T> [[nodiscard]] auto range(Extents<T> extents) noexcept
    {
        return static_cast<T>(extents.lerp(this->random()));
    }

    [[nodiscard]] auto vector(const BoundingBox<f64>& bounding_box) noexcept -> Vector2;

    [[nodiscard]] auto polar_vector(Extents<f64> radius_range,
                                    Extents<f64> angle_range = {0.0, 2 * math::pi}) noexcept
        -> Vector2;

    [[nodiscard]] auto choice(const ranges::range auto& iterable)
    {
        return *(iterable.begin() + static_cast<u64>(this->random() * iterable.size()));
    }

    template <typename T> [[nodiscard]] auto choice(std::initializer_list<T> init_list)
    {
        return *(init_list.begin() + static_cast<u64>(this->random() * init_list.size()));
    }

    [[nodiscard]] auto poisson_disc_points(f64 radius,
                                           Vector2 sample_region_size,
                                           u64 sample_count = 30UL) -> std::vector<Vector2>;

    [[nodiscard]] auto boolean(f64 threshold = 0.5) -> bool;

    [[nodiscard]] auto gaussian(f64 mean, f64 deviation = 1.0) -> f64;
};
} // namespace sm
