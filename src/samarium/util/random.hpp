/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>
#include <random>

#include "samarium/math/interp.hpp"

namespace sm::random
{
static auto cache_length = size_t{1000};

static auto current = size_t{0};

namespace detail
{
static auto get_random(size_t size)
{
    // std::random_device rand_dev;
    std::vector<f64> values(size);

    std::mt19937 generator(247);
    std::uniform_real_distribution<f64> distribution(0., 1.);
    std::generate(values.begin(), values.end(),
                  [&] { return distribution(generator); });
    return values;
}
} // namespace detail

static std::vector<f64> cache = detail::get_random(4000u);

auto fill_cache(size_t size) { cache = std::move(detail::get_random(size)); }

auto random() { return cache[current++ % cache_length]; }

template <typename T> [[nodiscard]] auto rand_range(Extents<T> range)
{
    return static_cast<T>(range.lerp(random()));
}

[[nodiscard]] auto rand_vector(Rect<f64> bounding_box)
{
    return Vector2{rand_range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   rand_range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] auto rand_vector(Extents<f64> radius_range,
                               Extents<f64> angle_range)
{
    return Vector2::from_polar(
        rand_range<f64>({radius_range.min, radius_range.max}),
        rand_range<f64>({angle_range.min, angle_range.max}));
}
} // namespace sm::random
