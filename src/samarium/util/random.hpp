/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <array>

#include "../math/BoundingBox.hpp"
#include "../math/interp.hpp"

namespace sm::random
{
class LinearCongruentialGenerator
{
  public:
    constexpr LinearCongruentialGenerator(u16 new_seed = 0) noexcept : seed{new_seed} {}

    [[nodiscard]] constexpr auto next() noexcept
    {
        seed = seed * a + c;
        return (seed >> 16) & 0x7fff;
    }

    u32 seed;

  private:
    const u16 a = u16(214013);
    const u16 c = u16(2531011);
};

template <size_t size> class Cache
{
    std::array<f64, size> cache;
    u64 current_index{};

  public:
    constexpr Cache() noexcept
    {
        auto rng = sm::random::LinearCongruentialGenerator{};
        std::generate(cache.begin(), cache.end(),
                      [&rng] { return static_cast<f64>(rng.next()) / 32768.0; });
    }

    [[nodiscard]] constexpr auto operator[](u32 index) noexcept { return cache[index]; }

    [[nodiscard]] constexpr auto next() noexcept
    {
        current_index++;
        return cache[current_index];
    }

    constexpr auto reseed(u32 new_seed) noexcept
    {
        auto rng = sm::random::LinearCongruentialGenerator{new_seed};
        std::generate(cache.begin(), cache.end(),
                      [&rng] { return static_cast<f64>(rng.next()) / 32768.0; });

        current_index = 0UL;
    }
};

static constinit auto cache = Cache<1024UL>{};

constexpr auto random() { return cache.next(); }

template <typename T> [[nodiscard]] constexpr auto rand_range(Extents<T> range_) noexcept
{
    return static_cast<T>(range_.lerp(random()));
}

[[nodiscard]] constexpr auto rand_vector(BoundingBox<f64> bounding_box) noexcept
{
    return Vector2{rand_range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   rand_range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] constexpr auto rand_vector(Extents<f64> radius_range,
                                         Extents<f64> angle_range) noexcept
{
    return Vector2::from_polar({.length = rand_range<f64>({radius_range.min, radius_range.max}),
                                .angle  = rand_range<f64>({angle_range.min, angle_range.max})});
}
} // namespace sm::random
