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
    std::vector<double> values(size);

    std::mt19937 generator(247);
    std::uniform_real_distribution<double_t> distribution(0., 1.);
    std::generate(values.begin(), values.end(),
                  [&] { return distribution(generator); });
    return values;
}
} // namespace detail

static std::vector<double> cache = detail::get_random(4000u);

auto fill_cache(size_t size) { cache = std::move(detail::get_random(size)); }

auto random() { return cache[current++ % cache_length]; }

template <typename T> [[nodiscard]] auto rand_range(Extents<T> range)
{
    return static_cast<T>(range.lerp(random()));
}

[[nodiscard]] auto rand_vector(Rect<double> bounding_box)
{
    return Vector2{
        rand_range<double_t>({bounding_box.min.x, bounding_box.max.x}),
        rand_range<double_t>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] auto rand_vector(Extents<double_t> radius_range,
                               Extents<double_t> angle_range)
{
    return Vector2::from_polar(
        rand_range<double_t>({radius_range.min, radius_range.max}),
        rand_range<double_t>({angle_range.min, angle_range.max}));
}
} // namespace sm::random
