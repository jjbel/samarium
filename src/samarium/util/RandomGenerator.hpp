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

        ranges::generate(cache,
                         [this] {
                             return static_cast<f64>(this->next()) /
                                    static_cast<f64>(std::numeric_limits<u32>::max());
                         });
    }

    void resize(u64 new_size);

    void reseed(u64 new_seed);

    [[nodiscard]] auto next() noexcept -> u64
    {
        const auto oldstate = this->state;
        // Advance internal state
        this->state = oldstate * magic_number + (this->inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        const auto xorshifted = static_cast<u32>(((oldstate >> 18UL) ^ oldstate) >> 27UL);
        const auto rot        = static_cast<u32>(oldstate >> 59UL);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

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


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_RANDOM_GENERATOR_IMPL)

#include <cmath>   // for ceil
#include <limits>  // for numeric_limits
#include <numbers> // for sqrt2
#include <tuple>   // for _Swallow_assign, ignore

#include "range/v3/algorithm/generate.hpp" // for generate, generate_fn

#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/range.hpp"       // for start_end
#include "samarium/math/vector_math.hpp" // for within_distance
#include "samarium/util/Grid.hpp"        // for Grid

namespace sm
{
namespace detail
{
[[nodiscard]] auto poisson_is_valid(Vector2 candidate,
                                    Vector2 sample_region,
                                    f64 cell_size,
                                    f64 radius,
                                    const std::vector<Vector2>& points,
                                    const Grid<i32>& grid)
{
    // TODO use BoundingBox
    if (!(candidate.x >= 0.0 && candidate.x < sample_region.x && candidate.y >= 0.0 &&
          candidate.y < sample_region.y))
    {
        return false;
    }

    const auto cell_x         = static_cast<i64>(candidate.x / cell_size);
    const auto cell_y         = static_cast<i64>(candidate.y / cell_size);
    const auto search_start_x = static_cast<u64>(math::max<i64>(0L, cell_x - 2L));
    const auto search_end_x =
        static_cast<u64>(math::min<i64>(cell_x + 2L, static_cast<i64>(grid.dims.x)));
    const auto search_start_y = static_cast<u64>(math::max<i64>(0L, cell_y - 2));
    const auto search_end_y =
        static_cast<u64>(math::min<i64>(cell_y + 2L, static_cast<i64>(grid.dims.y)));

    for (auto x : range::start_end(search_start_x, search_end_x))
    {
        for (auto y : range::start_end(search_start_y, search_end_y))
        {
            const auto point_index = grid[Indices{x, y}] - 1;
            if (point_index != -1 &&
                math::within_distance(candidate, points[static_cast<u64>(point_index)], radius))
            {
                return false;
            }
        }
    }
    return true;
}
} // namespace detail

void RandomGenerator::resize(u64 new_size)
{
    if (new_size < cache.size()) { return; }
    cache.resize(new_size);
    ranges::generate(cache, [this] { return this->next_scaled(); });
}

void RandomGenerator::reseed(u64 new_seed)
{
    inc           = new_seed;
    current_index = 0UL;
    ranges::generate(cache, [this] { return this->next_scaled(); });
}

[[nodiscard]] auto RandomGenerator::next_scaled() noexcept -> f64
{
    return static_cast<f64>(this->next()) / static_cast<f64>(std::numeric_limits<u32>::max());
}

[[nodiscard]] auto RandomGenerator::random() -> f64
{
    if (current_index < cache.size()) { return cache[current_index++]; }

    return this->next_scaled();
}

[[nodiscard]] auto RandomGenerator::operator()() -> f64 { return this->random(); }

[[nodiscard]] auto RandomGenerator::vector(const BoundingBox<f64>& bounding_box) noexcept -> Vector2
{
    return Vector2{this->range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   this->range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] auto RandomGenerator::polar_vector(Extents<f64> radius_range,
                                                 Extents<f64> angle_range) noexcept -> Vector2
{
    return Vector2::from_polar({.length = this->range<f64>({radius_range.min, radius_range.max}),
                                .angle  = this->range<f64>({angle_range.min, angle_range.max})});
}

[[nodiscard]] auto RandomGenerator::poisson_disc_points(f64 radius,
                                                        Vector2 sample_region_size,
                                                        u64 sample_count) -> std::vector<Vector2>
{
    const f64 cell_size = radius / std::numbers::sqrt2;

    auto grid = Grid<i32>{{static_cast<u64>(std::ceil(sample_region_size.x / cell_size)),
                           static_cast<u64>(std::ceil(sample_region_size.y / cell_size))}};

    auto points       = std::vector<Vector2>();
    auto spawn_points = std::vector<Vector2>();

    spawn_points.push_back(sample_region_size / 2.0);

    while (!spawn_points.empty())
    {
        const auto spawn_index  = this->range<u64>({0UL, spawn_points.size() - 1UL});
        auto spawn_centre       = spawn_points[spawn_index];
        auto candidate_accepted = false;

        for (auto i : range::end(sample_count))
        {
            std::ignore          = i;
            const auto candidate = spawn_centre + this->polar_vector({radius, 2 * radius});

            if (detail::poisson_is_valid(candidate, sample_region_size, cell_size, radius, points,
                                         grid))
            {
                points.push_back(candidate);
                spawn_points.push_back(candidate);
                grid[{static_cast<u64>(candidate.x / cell_size),
                      static_cast<u64>(candidate.y / cell_size)}] = static_cast<i32>(points.size());

                candidate_accepted = true;
                break;
            }
        }

        if (!candidate_accepted)
        {
            spawn_points.erase(spawn_points.begin() + static_cast<i64>(spawn_index));
        }
    }

    return points;
}

[[nodiscard]] auto RandomGenerator::boolean(f64 threshold) -> bool
{
    return this->random() < threshold;
}

[[nodiscard]] auto RandomGenerator::gaussian(f64 mean, f64 deviation) -> f64
{
    // from https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Implementation
    constexpr auto epsilon = std::numeric_limits<f64>::epsilon();
    // create two random numbers, make sure u1 is greater than epsilon
    auto u1 = 0.0;
    do {
        u1 = this->random();
    } while (u1 <= epsilon);
    const auto u2 = this->random();

    const auto mag = deviation * sqrt(-2.0 * std::log(u1));
    return mag * std::cos(math::two_pi * u2) + mean;
}
} // namespace sm
#endif
