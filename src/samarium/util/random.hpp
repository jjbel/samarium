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
constexpr static auto cache_length = size_t{100};

static auto current = size_t{0};

const static auto cache = []()
{
    auto values = std::array<double_t, cache_length>{};
    // std::random_device rand_dev;
    std::mt19937 generator(247);
    std::uniform_real_distribution<double_t> distribution(0., 1.);
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
    return values;
}();

auto random() { return cache[current++ % cache_length]; }

template <typename T> [[nodiscard]] auto rand_range(Extents<T> range)
{
    return static_cast<T>(range.lerp(random()));
}
} // namespace sm::random
