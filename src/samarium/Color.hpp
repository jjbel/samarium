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
#include <concepts>
#include <cstdint>
#include <string_view>

#include "fmt/format.h"

#include "math.hpp"
#include "types.hpp"
#include "util.hpp"

namespace sm
{
constexpr inline class RGB_t
{
} RGB{};
constexpr inline class RGBA_t
{
} RGBA{};
constexpr inline class BGR_t
{
} BGR{};
constexpr inline class BGRA_t
{
} BGRA{};
constexpr inline class hex_t
{
} hex{};


class Color
{
  public:
    u8 r;
    u8 g;
    u8 b;
    u8 a;

    constexpr Color() noexcept : r{}, g{}, b{}, a{ 255u } {}

    constexpr Color(u8 red, u8 green, u8 blue) noexcept
        : r(red), g(green), b(blue), a(255u)
    {
    }

    constexpr Color(u8 red, u8 green, u8 blue, u8 alpha) noexcept
        : r(red), g(green), b(blue), a(alpha)
    {
    }

    // constexpr Color(uint32_t n) noexcept
    //     : r(static_cast<u8>((n >> 24) & 0xFF)),
    //       g(static_cast<u8>((n >> 16) & 0xFF)),
    //       b(static_cast<u8>((n >> 8) & 0xFF)), a(static_cast<u8>(n & 0xFF))
    // {
    // }

    consteval Color(const char* str)
    {
        const auto length = util::strlen(str);
        if (str[0] != '#') throw std::logic_error("Hex string must start with #");
        if (length != 7 && length != 9)
            throw std::logic_error("Hex string must be 7 or 9 characters long");

        this->r = static_cast<u8>(16 * util::hex_to_int_safe(str[1]) +
                                       util::hex_to_int_safe(str[2]));
        this->g = static_cast<u8>(16 * util::hex_to_int_safe(str[3]) +
                                       util::hex_to_int_safe(str[4]));
        this->b = static_cast<u8>(16 * util::hex_to_int_safe(str[5]) +
                                       util::hex_to_int_safe(str[6]));

        if (length == 7) this->a = 255u;
        else
            this->a = static_cast<u8>(16 * util::hex_to_int_safe(str[7]) +
                                           util::hex_to_int_safe(str[8]));
    }

    // https://en.m.wikipedia.org/wiki/Alpha_compositing
    constexpr auto add_alpha_over(Color that) noexcept
    {
        // const double under     = this->a / 255.0;
        // const double over      = that.a / 255.0;
        // const double new_alpha = over + under * (1. - over);
        // this->r = static_cast<u8>(that.r * over + this->r * under * (1. - over));
        // this->g = static_cast<u8>(that.g * over + this->g * under * (1. - over));
        // this->b = static_cast<u8>(that.b * over + this->b * under * (1. - over));
        // this->a = static_cast<u8>(new_alpha * 255.);
        const auto alpha = 1.0 / 255 * that.a;
        r = static_cast<u8>(that.a / 255.0 * that.r + (1.0 - alpha) * r);
        g = static_cast<u8>(that.a / 255.0 * that.g + (1.0 - alpha) * g);
        b = static_cast<u8>(that.a / 255.0 * that.b + (1.0 - alpha) * b);
        a = static_cast<u8>((a / 255.0 + (1.0 - a / 255.0) * (alpha)) * 255);
    }

    template <util::integral T = u8>
    auto data(RGB_t /* color_format */) const noexcept
    {
        return std::array{ static_cast<T>(this->r), static_cast<T>(this->g),
                           static_cast<T>(this->b) };
    }

    template <util::integral T = u8>
    auto data(RGBA_t /* color_format */) const noexcept
    {
        return std::array{ static_cast<T>(this->r), static_cast<T>(this->g),
                           static_cast<T>(this->b), static_cast<T>(this->a) };
    }

    template <util::integral T = u8>
    auto data(BGR_t /* color_format */) const noexcept
    {
        return std::array{ static_cast<T>(b), static_cast<T>(g), static_cast<T>(r) };
    }

    template <util::integral T = u8>
    auto data(BGRA_t /* color_format */) const noexcept
    {
        return std::array{ static_cast<T>(b), static_cast<T>(g), static_cast<T>(r),
                           static_cast<T>(a) };
    }
};

[[nodiscard]] constexpr inline bool operator==(const Color& lhs,
                                               const Color& rhs) noexcept
{
    return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b && lhs.a == rhs.a;
}

namespace literals
{
template <Color color> consteval auto operator""_c() { return color; }

} // namespace literals


template <typename T>
concept ColorFormat = util::IsAnyOf<T, RGB_t, RGBA_t, BGR_t, BGRA_t>;
} // namespace sm


template <> class fmt::formatter<sm::Color>
{
  public:
    constexpr inline auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext> auto format(const sm::Color& p, FormatContext& ctx)
    {
        return format_to(ctx.out(),
                         "\x1b[38;2;{0};{1};{2}mCol\x1b[0m[{0}, {1}, {2}, {3}]", p.r, p.g,
                         p.b, p.a);
    }
};
