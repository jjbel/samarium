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

#include "samarium/core/types.hpp"
#include "samarium/math/math.hpp"
#include "samarium/util/util.hpp"

namespace sm
{
struct RGB_t
{
    constexpr static auto length = size_t{3};
};
struct RGBA_t
{
    constexpr static auto length = size_t{4};
};
struct BGR_t
{
    constexpr static auto length = size_t{3};
};
struct BGRA_t
{
    constexpr static auto length = size_t{4};
};

constexpr inline auto rgb  = RGB_t{};
constexpr inline auto rgba = RGBA_t{};
constexpr inline auto bgr  = BGR_t{};
constexpr inline auto bgra = BGRA_t{};

class Color
{
  public:
    u8 r{};
    u8 g{};
    u8 b{};
    u8 a{255u};

    static consteval auto from_hex(const char* str)
    {
        const auto length = util::strlen(str);
        if (str[0] != '#') throw std::logic_error("Hex string must start with #");
        if (length != 7u && length != 9u)
            throw std::logic_error("Hex string must be 7 or 9 characters long");

        const auto r = static_cast<u8>(16 * util::hex_to_int_safe(str[1]) +
                                       util::hex_to_int_safe(str[2]));
        const auto g = static_cast<u8>(16 * util::hex_to_int_safe(str[3]) +
                                       util::hex_to_int_safe(str[4]));
        const auto b = static_cast<u8>(16 * util::hex_to_int_safe(str[5]) +
                                       util::hex_to_int_safe(str[6]));

        const auto a = length == 7u ? u8{255}
                                    : static_cast<u8>(16 * util::hex_to_int_safe(str[7]) +
                                                      util::hex_to_int_safe(str[8]));
        return Color{r, g, b, a};
    }

    // https://en.m.wikipedia.org/wiki/Alpha_compositing

    constexpr auto add_alpha_over(const Color& that) noexcept
    {
        const auto alpha = 1.0 / 255 * that.a;
        r                = static_cast<u8>(that.a / 255.0 * that.r + (1.0 - alpha) * r);
        g                = static_cast<u8>(that.a / 255.0 * that.g + (1.0 - alpha) * g);
        b                = static_cast<u8>(that.a / 255.0 * that.b + (1.0 - alpha) * b);
        a                = static_cast<u8>((a / 255.0 + (1.0 - a / 255.0) * alpha) * 255);
    }

    [[nodiscard]] constexpr auto with_alpha(u8 alpha) const
    {
        return Color{r, g, b, alpha};
    }

    [[nodiscard]] constexpr auto with_multiplied_alpha(double factor) const
    {
        return Color{r, g, b, static_cast<u8>(a * factor)};
    }

    template <concepts::integral T = u8>
    [[nodiscard]] auto get_formatted(RGB_t /* color_format */) const noexcept
    {
        return std::array{static_cast<T>(this->r), static_cast<T>(this->g),
                          static_cast<T>(this->b)};
    }

    template <concepts::integral T = u8>
    [[nodiscard]] auto get_formatted(RGBA_t /* color_format */) const noexcept
    {
        return std::array{static_cast<T>(this->r), static_cast<T>(this->g),
                          static_cast<T>(this->b), static_cast<T>(this->a)};
    }

    template <concepts::integral T = u8>
    [[nodiscard]] auto get_formatted(BGR_t /* color_format */) const noexcept
    {
        return std::array{static_cast<T>(b), static_cast<T>(g), static_cast<T>(r)};
    }

    template <concepts::integral T = u8>
    [[nodiscard]] auto get_formatted(BGRA_t /* color_format */) const noexcept
    {
        return std::array{static_cast<T>(b), static_cast<T>(g), static_cast<T>(r),
                          static_cast<T>(a)};
    }

    [[nodiscard]] constexpr friend bool operator==(const Color& lhs,
                                                   const Color& rhs) noexcept = default;
};

namespace literals
{
consteval auto operator""_c(const char* str, size_t) { return Color::from_hex(str); }

} // namespace literals

template <typename T>
concept color_format_concept = concepts::is_any_of<T, RGB_t, RGBA_t, BGR_t, BGRA_t>;
} // namespace sm


template <> class fmt::formatter<sm::Color>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const sm::Color& p, FormatContext& ctx) //
    {
        return format_to(ctx.out(),
                         "\x1b[38;2;{0};{1};{2}mCol\x1b[0m[{0}, {1}, {2}, {3}]", p.r, p.g,
                         p.b, p.a);
    }
};
