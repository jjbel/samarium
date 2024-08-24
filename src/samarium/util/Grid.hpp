/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>      // for span
#include <stdexcept> // for out_of_range
#include <vector>    // for vector

#include "fmt/format.h"

#include "range/v3/algorithm/copy.hpp"
#include "range/v3/view/enumerate.hpp"
#include "range/v3/view/iota.hpp"
#include "range/v3/view/transform.hpp"
#include "range/v3/view/zip.hpp"

#include "samarium/graphics/Color.hpp"   // for Color
#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/loop.hpp"        // for end

namespace sm
{
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

    std::vector<T> elements;
    const Dimensions dims;

    // Constructors
    explicit Grid(Dimensions dims_) : elements(dims_.x * dims_.y), dims{dims_} {}

    Grid(Dimensions dims_, T init_value) : elements(dims_.x * dims_.y, init_value), dims{dims_} {}

    explicit Grid(std::span<const T> span, Dimensions dims_)
        : elements(span.begin(), span.end()), dims{dims_}
    {
    }

    template <typename Fn> static auto generate(Dimensions dims, Fn&& fn)
    {
        auto grid = Grid<T>(dims);
        for (auto y : loop::end(dims.y))
        {
            for (auto x : loop::end(dims.x))
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
        return this->elements[indices.y * this->dims.x + indices.x];
    }
    auto operator[](Indices indices) const -> const_reference
    {
        return this->elements[indices.y * this->dims.x + indices.x];
    }

    auto operator[](u64 index) noexcept -> T& { return this->elements[index]; }
    auto operator[](u64 index) const noexcept -> const_reference { return this->elements[index]; }

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
        else [[likely]] { return this->elements[index]; }
    }

    auto at(u64 index) const -> const_reference
    {
        if (index >= this->size()) [[unlikely]]
        {
            throw std::out_of_range(
                fmt::format("sm::Grid: index {} out of range for size {}", index, this->size()));
        }
        else [[likely]] { return this->elements[index]; }
    }

    auto at_or(Indices indices, T default_value) const -> T
    {
        if (indices.x >= dims.x || indices.y >= dims.y) [[unlikely]] { return default_value; }
        else [[likely]] { return this->operator[](indices); }
    }

    auto at_or(u64 index, T default_value) const -> T
    {
        if (index >= this->size()) [[unlikely]] { return default_value; }
        else [[likely]] { return this->elements[index]; }
    }

    auto begin() { return this->elements.begin(); }
    auto end() { return this->elements.end(); }

    auto begin() const { return this->elements.cbegin(); }
    auto end() const { return this->elements.cend(); }

    auto cbegin() const { return this->elements.cbegin(); }
    auto cend() const { return this->elements.cend(); }

    auto front() const -> const_reference { return elements.front(); }
    auto front() -> reference { return elements.front(); }

    auto back() const -> const_reference { return elements.back(); }
    auto back() -> reference { return elements.back(); }

    auto size() const { return this->elements.size(); }
    auto empty() const { return this->elements.size() == 0; }

    auto bounding_box() const { return BoundingBox<u64>{Indices{}, dims - Indices{1, 1}}; }

    auto fill(const T& value) { this->elements.fill(value); }

    template <concepts::ColorFormat Format> [[nodiscard]] auto formatted_data(Format format) const
    {
        const auto format_length = Format::length;
        auto output              = std::vector<std::array<u8, format_length>>(this->size());
        const auto converter     = [format](auto color) { return color.get_formatted(format); };

        ranges::copy(ranges::views::transform(this->elements, converter), output.begin());
        // std::transform(this->data.cbegin(), this->data.cend(), output.begin(), converter);

        return output;
    }

    [[nodiscard]] auto span() { return std::span{elements}; }

    [[nodiscard]] iterator data() { return elements.data(); }

    [[nodiscard]] const_iterator data() const { return elements.data(); }

    [[nodiscard]] auto upscale(u64 upscale_factor) const
    {
        auto output = Grid<T>(this->dims * upscale_factor);
        for (auto y : loop::end(output.dims.y))
        {
            for (auto x : loop::end(output.dims.x))
            {
                output[{x, y}] = this->operator[](Indices{x, y} / upscale_factor);
            }
        }

        return output;
    }

    auto enumerate_1d() { return ranges::views::enumerate(elements); }

    auto enumerate_2d() { return ranges::views::zip(iota_view_2d(dims), elements); }

    auto byte_size() const { return size() * sizeof(T); }
};

using Image = Grid<Color>;

// TODO the fields should have their own transform and rounding, to make indexing easier.
// see flow_field_noise
using ScalarField = Grid<f64>;
using VectorField = Grid<Vector2>;

constexpr static auto dims4K  = Dimensions{3840UL, 2160UL};
constexpr static auto dimsFHD = Dimensions{1920UL, 1080UL};
constexpr static auto dims720 = Dimensions{1280UL, 720UL};
constexpr static auto dims480 = Dimensions{640UL, 480UL};
constexpr static auto dimsP2  = Dimensions{2048UL, 1024UL};
} // namespace sm
