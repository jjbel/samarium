/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../core/DynArray.hpp"
#include "../math/Rect.hpp"
#include "../util/print.hpp"

#include "Color.hpp"

namespace sm
{
constexpr inline auto dims4K  = Dimensions{3840u, 2160u};
constexpr inline auto dims720 = Dimensions{1280u, 720u};
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

template <typename T> class Grid
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
    DynArray<T> data;
    const Dimensions dims;

    // Constructors
    explicit Grid(Dimensions dims_ = dimsFHD) : data(dims_.x * dims_.y), dims{dims_} {}

    explicit Grid(Dimensions dims_, T init_value) : data(dims_.x * dims_.y, init_value), dims{dims_}
    {
    }

    template <typename Function>
    requires std::invocable<Function, Indices>
    static auto generate(Dimensions dims, Function&& fn)
    {
        Grid<T> grid(dims);
        const auto beg = grid.begin();
        std::for_each(beg, grid.end(),
                      [fn = std::forward<Function>(fn), beg](auto& element)
                      { element = fn(convert_1d_to_2d(&element - beg)); });
    }

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

    template <concepts::ColorFormat Format>
    DynArray<std::array<u8, Format::length>> formatted_data(Format format) const
    {
        const auto format_length = Format::length;
        auto fmt_data            = DynArray<std::array<u8, format_length>>(this->size());

        std::transform(std::execution::par_unseq, this->begin(), this->end(), fmt_data.begin(),
                       [format](auto color) { return color.get_formatted(format); });

        return fmt_data;
    }
};

using Image = Grid<Color>;
} // namespace sm
