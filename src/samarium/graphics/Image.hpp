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

#include <algorithm>
#include <concepts>
#include <iterator>
#include <memory>
#include <utility>

#include "samarium/core/DynArray.hpp"
#include "samarium/math/Rect.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/util/print.hpp"

#include "Color.hpp"

namespace sm
{
constexpr inline auto dims4K  = Dimensions{3840u, 2160u};
constexpr inline auto dimsHD  = Dimensions{1280u, 720u};
constexpr inline auto dimsFHD = Dimensions{1920u, 1080u};
constexpr inline auto dimsP2  = Dimensions{2048u, 1024u};

constexpr auto convert_1d_to_2d(Dimensions dims, size_t index)
{
    return Indices{index % dims.x, index / dims.x};
}

constexpr auto convert_2d_to_1d(Dimensions dims, Indices coordinates)
{
    return coordinates.y * dims.x + coordinates.x;
}

template <typename T> class Field
{
  public:
    // Container types
    using value_type      = T;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = T const*;
    using difference_type = std::ptrdiff_t;
    using size_type       = std::size_t;

    // Public members
    const Dimensions dims;

    DynArray<T> data;

    // Constructors
    Field(Dimensions dims_ = dimsFHD) : dims(dims_), data(dims.x * dims.y) {}

    Field(Dimensions dims_, T init_value) : dims(dims_), data(dims.x * dims.y, init_value) {}

    // Member functions
    T& operator[](Indices indices) { return this->data[indices.y * this->dims.x + indices.x]; }

    const T& operator[](Indices indices) const
    {
        return this->data[indices.y * this->dims.x + indices.x];
    }

    T& operator[](size_t index) { return this->data[index]; }

    const T& operator[](size_t index) const { return this->data[index]; }

    auto begin() { return this->data.begin(); }
    auto end() { return this->data.end(); }

    auto begin() const { return this->data.cbegin(); }
    auto end() const { return this->data.cend(); }

    auto cbegin() const { return this->data.cbegin(); }
    auto cend() const { return this->data.cend(); }

    auto size() const { return this->data.size(); }
    auto max_size() const { return this->data.size(); } // for stl compatibility
    auto empty() const { return this->data.size() == 0; }

    auto rect() const { return Rect<size_t>{Indices{}, dims - Indices{1, 1}}; }

    auto fill(const T& value) { this->data.fill(value); }

    template <color_format_concept Format>
    DynArray<std::array<u8, Format::length>> formatted_data(Format format) const
    {
        const auto format_length = Format::length;
        auto fmt_data            = DynArray<std::array<u8, format_length>>(this->size());

        std::transform(std::execution::par_unseq, this->begin(), this->end(), fmt_data.begin(),
                       [format](auto color) { return color.get_formatted(format); });

        return fmt_data;
    }
};

using Image = Field<Color>;

// Since data is already stored as RGBA, no need to convert it, directly return it
// template <> template <> inline auto Image::formatted_data(RGBA_t) const
// {
//     return std::span{this->data.cbegin(), this->size() * 4};
// }
} // namespace sm
