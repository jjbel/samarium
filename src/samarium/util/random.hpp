/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

#include "../math/BoundingBox.hpp"
#include "../math/interp.hpp"
#include "samarium/graphics/Image.hpp"
#include "samarium/math/Vector2.hpp"

// clang doesn't properly support constinit yet:
#ifndef __clang__
#define SM_CUSTOM_CONSTEXPR constexpr
#else
#define SM_CUSTOM_CONSTEXPR inline
#endif // __clang__

#ifndef __clang__
#define SM_CUSTOM_CONSTINIT constinit
#else
#define SM_CUSTOM_CONSTINIT
#endif // __clang__

namespace sm::random
{
// PCG-based random number generator, see https://www.pcg-random.org/
struct Generator
{
    static constexpr auto magic_number = 6364136223846793005ULL;

    u64 state;
    u64 inc;

    constexpr Generator(u64 new_state = 69, u64 new_inc = 69) noexcept
        : state{new_state * magic_number + (new_inc | 1)}, inc{new_inc}
    {
    }

    [[nodiscard]] constexpr auto next() noexcept
    {
        const auto oldstate = this->state;
        // Advance internal state
        this->state = oldstate * magic_number + (this->inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        const auto xorshifted = static_cast<u32>(((oldstate >> 18UL) ^ oldstate) >> 27UL);
        const auto rot        = static_cast<u32>(oldstate >> 59UL);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    [[nodiscard]] constexpr auto next_scaled() noexcept
    {
        return static_cast<f64>(this->next()) / static_cast<f64>(std::numeric_limits<u32>::max());
    }
};


template <size_t Size> class Cache
{
    std::array<f64, Size> cache;
    u64 current_index{};

  public:
    SM_CUSTOM_CONSTEXPR Cache() noexcept
    {
        auto rng = Generator{};
        std::generate(cache.begin(), cache.end(), [&rng] { return rng.next_scaled(); });
    }

    [[nodiscard]] SM_CUSTOM_CONSTEXPR auto operator[](u32 index) noexcept { return cache[index]; }

    [[nodiscard]] SM_CUSTOM_CONSTEXPR auto next() noexcept
    {
        current_index++;
        return cache[current_index];
    }

    SM_CUSTOM_CONSTEXPR auto reseed(u16 new_seed) noexcept
    {
        auto rng = Generator{new_seed};
        std::generate(cache.begin(), cache.end(), [&rng] { return rng.next_scaled(); });

        current_index = 0UL;
    }
};

static SM_CUSTOM_CONSTINIT auto cache = Cache<65536UL>{};

SM_CUSTOM_CONSTEXPR auto random() { return cache.next(); }

template <typename T> [[nodiscard]] SM_CUSTOM_CONSTEXPR auto in_range(Extents<T> extents) noexcept
{
    return static_cast<T>(extents.lerp(random()));
}

[[nodiscard]] SM_CUSTOM_CONSTEXPR auto vector(const BoundingBox<f64>& bounding_box) noexcept
{
    return Vector2{in_range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   in_range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] SM_CUSTOM_CONSTEXPR auto vector(Extents<f64> radius_range,
                                              Extents<f64> angle_range) noexcept
{
    return Vector2::from_polar({.length = in_range<f64>({radius_range.min, radius_range.max}),
                                .angle  = in_range<f64>({angle_range.min, angle_range.max})});
}

[[nodiscard]] SM_CUSTOM_CONSTEXPR auto choice(const concepts::Range auto& iterable)
{
    return *(iterable.begin() + static_cast<u64>(random() * iterable.size()));
}

template <typename T>
[[nodiscard]] SM_CUSTOM_CONSTEXPR auto choice(std::initializer_list<T> init_list)
{
    return *(init_list.begin() + static_cast<u64>(random() * init_list.size()));
}
} // namespace sm::random
