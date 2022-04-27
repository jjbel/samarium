/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <initializer_list>
#include <vector>

#include "../graphics/Image.hpp"
#include "../math/BoundingBox.hpp"

namespace sm
{
// PCG-based random number generator, see https://www.pcg-random.org/
struct RandomGenerator
{
    static constexpr auto magic_number = 6364136223846793005ULL;

    std::vector<f64> cache;
    u64 state;
    u64 inc;
    u64 current_index{};

    RandomGenerator(u64 cache_size = 1024UL, u64 new_state = 69, u64 new_inc = 69) noexcept
        : cache(cache_size), state{new_state * magic_number + (new_inc | 1)}, inc{new_inc}
    {
        std::generate(cache.begin(), cache.end(), [this] { return this->next_scaled(); });
    }

    void resize(u64 new_size);

    void reseed(u64 new_seed);

    [[nodiscard]] auto next() noexcept -> u64;

    [[nodiscard]] auto next_scaled() noexcept -> f64;

    [[nodiscard]] auto random() -> f64;

    template <typename T> [[nodiscard]] auto range(Extents<T> extents) noexcept
    {
        return static_cast<T>(extents.lerp(random()));
    }

    [[nodiscard]] Vector2 vector(const BoundingBox<f64>& bounding_box) noexcept;

    [[nodiscard]] Vector2 polar_vector(Extents<f64> radius_range,
                                       Extents<f64> angle_range = {0.0,
                                                                   2 * std::numbers::pi}) noexcept;

    [[nodiscard]] auto choice(const concepts::Range auto& iterable)
    {
        return *(iterable.begin() + static_cast<u64>(random() * iterable.size()));
    }

    template <typename T> [[nodiscard]] auto choice(std::initializer_list<T> init_list)
    {
        return *(init_list.begin() + static_cast<u64>(random() * init_list.size()));
    }

    [[nodiscard]] auto poisson_disc_points(f64 radius,
                                           Vector2 sample_region_size,
                                           u64 sample_count = 30UL) -> std::vector<Vector2>;
};

using Image       = Grid<Color>;
using ScalarField = Grid<f64>;
using VectorField = Grid<Vector2>;
} // namespace sm
