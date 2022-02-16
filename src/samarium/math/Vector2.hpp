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

#include "fmt/format.h"

#include "math.hpp"

namespace sm
{
template <concepts::number T> class Vector2_t
{
  public:
    T x{};
    T y{};

    constexpr auto length() const noexcept { return std::sqrt(x * x + y * y); }
    constexpr auto length_sq() const noexcept { return x * x + y * y; }
    constexpr auto angle() const noexcept { return std::atan2(y, x); }

    constexpr auto operator+=(const Vector2_t& rhs) noexcept
    {
        this->x += rhs.x;
        this->y += rhs.y;
        return *this;
    }

    constexpr auto operator-=(const Vector2_t& rhs) noexcept
    {
        this->x -= rhs.x;
        this->y -= rhs.y;
        return *this;
    }

    constexpr auto operator*=(const Vector2_t& rhs) noexcept
    {
        this->x *= rhs.x;
        this->y *= rhs.y;
        return *this;
    }

    constexpr auto operator*=(T rhs) noexcept
    {
        this->x *= rhs;
        this->y *= rhs;
        return *this;
    }

    constexpr auto operator/=(const Vector2_t& rhs) noexcept
    {
        this->x /= rhs.x;
        this->y /= rhs.y;
        return *this;
    }

    constexpr auto operator/=(T rhs) noexcept
    {
        this->x /= rhs;
        this->y /= rhs;
        return *this;
    }

    template <concepts::number U> constexpr auto as() const
    {
        return Vector2_t<U>{static_cast<U>(this->x), static_cast<U>(this->y)};
    }
    template <concepts::number U> constexpr operator Vector2_t<U>() const
    {
        return this->template as<U>();
    }

    [[nodiscard]] static constexpr auto dot(Vector2_t<T> p1, Vector2_t<T> p2)
    {
        return p1.x * p2.x + p1.y * p2.y;
    }
};

template <concepts::floating_point T>
[[nodiscard]] constexpr bool operator==(const Vector2_t<T>& lhs,
                                        const Vector2_t<T>& rhs) noexcept
{
    return math::almost_equal(lhs.x, rhs.x) && math::almost_equal(lhs.y, rhs.y);
}

template <concepts::integral T>
[[nodiscard]] constexpr bool operator==(const Vector2_t<T>& lhs,
                                        const Vector2_t<T>& rhs) noexcept
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template <concepts::number T>
[[nodiscard]] constexpr bool operator!=(const Vector2_t<T>& lhs, const Vector2_t<T>& rhs)
{
    return !operator==(lhs, rhs);
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator+(Vector2_t<T> lhs, const Vector2_t<T>& rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator-(Vector2_t<T> lhs, const Vector2_t<T>& rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator*(Vector2_t<T> lhs, const Vector2_t<T>& rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator*(Vector2_t<T> lhs, T rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator*(T lhs, Vector2_t<T> rhs) noexcept
{
    rhs *= lhs;
    return rhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator/(Vector2_t<T> lhs, const Vector2_t<T>& rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

template <concepts::number T>
[[nodiscard]] constexpr auto operator/(Vector2_t<T> lhs, T rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

using Vector2    = Vector2_t<double>;
using Indices    = Vector2_t<size_t>;
using Dimensions = Vector2_t<size_t>;

namespace literals
{
consteval auto operator"" _x(long double x) { return Vector2{static_cast<double>(x), 0}; }
consteval auto operator"" _y(long double y) { return Vector2{0, static_cast<double>(y)}; }
} // namespace literals
} // namespace sm

template <sm::concepts::number T> class fmt::formatter<sm::Vector2_t<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const sm::Vector2_t<T>& p, FormatContext& ctx)
    {
        return format_to(
            ctx.out(),
            (std::is_floating_point<T>::value ? "Vec({:6.2f}, {:6.2f})" : "Vec({}, {})"),
            p.x, p.y);
    }
};
