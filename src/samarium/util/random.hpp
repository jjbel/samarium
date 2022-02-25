/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
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
