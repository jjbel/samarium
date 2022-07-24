/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>
#include <functional>
#include <iterator>

#include "fmt/format.h"
#include "range/v3/algorithm/copy.hpp"
#include "range/v3/view/enumerate.hpp"
#include "range/v3/view/iota.hpp"
#include "range/v3/view/transform.hpp"
#include "range/v3/view/zip.hpp"

#include "../math/BoundingBox.hpp"
#include "../math/Extents.hpp"

#include "Color.hpp"

namespace sm
{
constexpr inline auto dims4K  = Dimensions{3840UL, 2160UL};
constexpr inline auto dimsFHD = Dimensions{1920UL, 1080UL};
constexpr inline auto dims720 = Dimensions{1280UL, 720UL};
constexpr inline auto dims480 = Dimensions{640UL, 480UL};
constexpr inline auto dimsP2  = Dimensions{2048UL, 1024UL};

constexpr auto convert_1d_to_2d(Dimensions dims, u64 index)
{
    return Indices{index % dims.x, index / dims.x};
}

constexpr auto convert_2d_to_1d(Dimensions dims, Indices coordinates)
{
    return coordinates.y * dims.x + coordinates.x;
}

inline auto iota_view_2d(Dimensions dims)
{
    return ranges::views::iota(0UL, dims.x * dims.y) |
           ranges::views::transform([dims](u64 index) { return convert_1d_to_2d(dims, index); });
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
    using size_type       = u64;

    // Public members
    std::vector<T> data;
    const Dimensions dims;

    // Constructors
    explicit Grid(Dimensions dims_ = dimsFHD) : data(dims_.x * dims_.y), dims{dims_} {}

    explicit Grid(Dimensions dims_, T init_value) : data(dims_.x * dims_.y, init_value), dims{dims_}
    {
    }

    template <typename Fn> static auto generate(Dimensions dims, Fn&& fn)
    {
        auto grid = Grid<T>(dims);
        for (auto y : range(dims.y))
        {
            for (auto x : range(dims.x))
            {
                const auto pos = Indices{x, y};
                grid[pos]      = fn(pos);
            }
        }
        return grid;
    }

    // -------------------------------Member functions------------------------------------

    auto operator[](Indices indices) -> reference
    {
        return this->data[indices.y * this->dims.x + indices.x];
    }
    auto operator[](Indices indices) const -> const_reference
    {
        return this->data[indices.y * this->dims.x + indices.x];
    }

    auto operator[](u64 index) noexcept -> T& { return this->data[index]; }
    auto operator[](u64 index) const noexcept -> const_reference { return this->data[index]; }

    auto at(Indices indices) -> T&
    {
        if (indices.x >= dims.x || indices.y >= dims.y) [[unlikely]]
        {
            throw std::out_of_range(
                fmt::format("sm::Grid: indices ({}, {}) out of range for dimensions ({}, {})",
                            indices.x, indices.y, this->dims.x, this->dims.y));
        }
        else [[likely]] { return this->operator[](indices); }
    }

    auto at(Indices indices) const -> const_reference
    {
        if (indices.x >= dims.x || indices.y >= dims.y) [[unlikely]]
        {
            throw std::out_of_range(
                fmt::format("sm::Grid: indices ({}, {}) out of range for dimensions ({}, {})",
                            indices.x, indices.y, this->dims.x, this->dims.y));
        }
        else [[likely]] { return this->operator[](indices); }
    }

    auto at(u64 index) -> T&
    {
        if (index >= this->size()) [[unlikely]]
        {
            throw std::out_of_range(
                fmt::format("sm::Grid: index {} out of range for size {}", index, this->size()));
        }
        else [[likely]] { return this->data[index]; }
    }

    auto at(u64 index) const -> const_reference
    {
        if (index >= this->size()) [[unlikely]]
        {
            throw std::out_of_range(
                fmt::format("sm::Grid: index {} out of range for size {}", index, this->size()));
        }
        else [[likely]] { return this->data[index]; }
    }

    auto at_or(Indices indices, T default_value) const -> T
    {
        if (indices.x >= dims.x || indices.y >= dims.y) [[unlikely]] { return default_value; }
        else [[likely]] { return this->operator[](indices); }
    }

    auto at_or(u64 index, T default_value) const -> T
    {
        if (index >= this->size()) [[unlikely]] { return default_value; }
        else [[likely]] { return this->data[index]; }
    }

    auto begin() { return this->data.begin(); }
    auto end() { return this->data.end(); }

    auto begin() const { return this->data.cbegin(); }
    auto end() const { return this->data.cend(); }

    auto cbegin() const { return this->data.cbegin(); }
    auto cend() const { return this->data.cend(); }

    auto front() const -> const_reference { return data.front(); }
    auto front() -> reference { return data.front(); }

    auto back() const -> const_reference { return data.back(); }
    auto back() -> reference { return data.back(); }

    auto size() const { return this->data.size(); }
    auto max_size() const { return this->data.size(); } // for stl compatibility
    auto empty() const { return this->data.size() == 0; }

    auto bounding_box() const { return BoundingBox<u64>{Indices{}, dims - Indices{1, 1}}; }

    auto fill(const T& value) { this->data.fill(value); }

    template <concepts::ColorFormat Format> [[nodiscard]] auto formatted_data(Format format) const
    {
        const auto format_length = Format::length;
        auto output              = std::vector<std::array<u8, format_length>>(this->size());
        const auto converter     = [format](auto color) { return color.get_formatted(format); };

        ranges::copy(ranges::views::transform(this->data, converter), output.begin());
        // std::transform(this->data.cbegin(), this->data.cend(), output.begin(), converter);

        return output;
    }

    [[nodiscard]] auto upscale(u64 upscale_factor) const
    {
        auto output = Grid<T>(this->dims * upscale_factor);
        for (auto y : range(output.dims.y))
        {
            for (auto x : range(output.dims.x))
            {
                output[{x, y}] = this->operator[](Indices{x, y} / upscale_factor);
            }
        }

        return output;
    }

    auto enumerate_1d() { return ranges::views::enumerate(*this); }

    auto enumerate_2d() { return ranges::views::zip(iota_view_2d(dims), *this); }
};
} // namespace sm
