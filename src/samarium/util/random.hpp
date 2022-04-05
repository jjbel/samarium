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

// clang doesn't properly support constinit yet:
#ifndef __clang__
#define SM_CUSTOM_CONSTEXPR constexpr
#else
#define SM_CUSTOM_CONSTEXPR
#endif // __clang__

#ifndef __clang__
#define SM_CUSTOM_CONSTINIT constinit
#else
#define SM_CUSTOM_CONSTINIT
#endif // __clang__

namespace sm::random
{
class LinearCongruentialGenerator
{
  public:
    SM_CUSTOM_CONSTEXPR explicit LinearCongruentialGenerator(u16 new_seed = 0) noexcept
        : seed{new_seed}
    {
    }

    [[nodiscard]] SM_CUSTOM_CONSTEXPR auto next() noexcept
    {
        seed                         = seed * a + c;
        constexpr auto magic_number0 = 16U;
        constexpr auto magic_number1 = 0x7fff;
        return (seed >> magic_number0) & magic_number1;
    }

  private:
    u32 seed;
    const u16 a = u16(214013);
    const u16 c = u16(2531011);
};

template <size_t Size> class Cache
{
    std::array<f64, Size> cache;
    u64 current_index{};

  public:
    SM_CUSTOM_CONSTEXPR Cache() noexcept
    {
        auto rng = sm::random::LinearCongruentialGenerator{};
        std::generate(cache.begin(), cache.end(),
                      [&rng] { return static_cast<f64>(rng.next()) / 32768.0; });
    }

    [[nodiscard]] SM_CUSTOM_CONSTEXPR auto operator[](u32 index) noexcept { return cache[index]; }

    [[nodiscard]] SM_CUSTOM_CONSTEXPR auto next() noexcept
    {
        current_index++;
        return cache[current_index];
    }

    SM_CUSTOM_CONSTEXPR auto reseed(u16 new_seed) noexcept
    {
        auto rng = sm::random::LinearCongruentialGenerator{new_seed};
        std::generate(cache.begin(), cache.end(),
                      [&rng] { return static_cast<f64>(rng.next()) / 32768.0; });

        current_index = 0UL;
    }
};

static SM_CUSTOM_CONSTINIT auto cache = Cache<1024UL>{};

SM_CUSTOM_CONSTEXPR inline auto random() { return cache.next(); }

template <typename T>
[[nodiscard]] SM_CUSTOM_CONSTEXPR inline auto rand_range(Extents<T> extents) noexcept
{
    return static_cast<T>(extents.lerp(random()));
}

[[nodiscard]] SM_CUSTOM_CONSTEXPR inline auto
rand_vector(const BoundingBox<f64>& bounding_box) noexcept
{
    return Vector2{rand_range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   rand_range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] SM_CUSTOM_CONSTEXPR inline auto rand_vector(Extents<f64> radius_range,
                                                          Extents<f64> angle_range) noexcept
{
    return Vector2::from_polar({.length = rand_range<f64>({radius_range.min, radius_range.max}),
                                .angle  = rand_range<f64>({angle_range.min, angle_range.max})});
}
} // namespace sm::random
