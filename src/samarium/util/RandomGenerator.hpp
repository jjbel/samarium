/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <initializer_list> // for initializer_list
#include <random>           // for random_device
#include <vector>           // for vector

#include "range/v3/algorithm/generate.hpp" // for generate, generate_fn
#include "range/v3/range/concepts.hpp"     // for random_access_range

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

// TODO unary minus operator applied to unsigned type, result still unsigned
#pragma warning(suppress : 4146)
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
                                    Extents<f64> angle_range = {0.0,
                                                                2 * math::pi}) noexcept -> Vector2;

    [[nodiscard]] auto choice(const ranges::range auto& iterable)
    {
        return *(iterable.begin() + static_cast<u64>(this->random() * iterable.size()));
    }

    template <typename T> [[nodiscard]] auto choice(std::initializer_list<T> init_list)
    {
        return *(init_list.begin() + static_cast<u64>(this->random() * init_list.size()));
    }

    // TODO remove this
    [[nodiscard]] auto poisson_disc_points(f64 radius,
                                           BoundingBox<f64> sample_region,
                                           u64 sample_count = 30UL) -> std::vector<Vector2>;

    [[nodiscard]] auto boolean(f64 threshold = 0.5) -> bool;

    // TODO rejection sampling for general dist
    [[nodiscard]] auto gaussian(f64 mean, f64 deviation = 1.0) -> f64;

    [[nodiscard]] auto choice(const ranges::random_access_range auto& iterable)
    {
        return iterable[static_cast<u64>(
            std::round(random() * static_cast<f64>(ranges::size(iterable) - 1)))];
    }
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_RANDOM_GENERATOR_IMPL)

#include <cmath>   // for ceil
#include <limits>  // for numeric_limits
#include <numbers> // for sqrt2
#include <tuple>   // for _Swallow_assign, ignore

#include "range/v3/algorithm/generate.hpp" // for generate, generate_fn

#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/loop.hpp"        // for start_end
#include "samarium/math/vector_math.hpp" // for within_distance
#include "samarium/util/Grid.hpp"        // for Grid

namespace sm
{
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
