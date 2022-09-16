/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "fmt/chrono.h"
#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "../core/version.hpp"
#include "../graphics/Color.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/Transform.hpp"
#include "../math/Vector2.hpp"
#include "../math/complex.hpp"
#include "../math/shapes.hpp"
#include "../physics/Particle.hpp"

namespace fmt
{
template <sm::concepts::Integral T> class formatter<sm::Vector2_t<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    constexpr auto format(sm::Vector2_t<T> p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "{}({:>3}, {:>3})",
                              fmt::styled("Vec", fmt::emphasis::bold), p.x, p.y);
    }
};

template <sm::concepts::FloatingPoint T> class formatter<sm::Vector2_t<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    constexpr auto format(sm::Vector2_t<T> p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "{}({:6.3f}, {:6.3f})",
                              fmt::styled("Vec", fmt::emphasis::bold), p.x, p.y);
    }
};

template <typename T, typename Char> struct is_tuple_formattable<sm::Vector2_t<T>, Char>
{
    static constexpr auto value = false;
};

template <> class formatter<sm::Version>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::Version& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "samarium version {}.{}.{}", p.major, p.minor, p.patch);
    }
};

template <> class formatter<sm::Particle>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::Particle& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "Particle(pos: {}, vel: {}, acc: {})", p.pos, p.acc,
                              p.vel);
    }
};

template <> class formatter<sm::Color>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::Color& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(),
                              "\x1b[38;2;{0};{1};{2}mCol\x1b[0m[{0:>3}, {1:>3}, {2:>3}, {3:>3}]",
                              p.r, p.g, p.b, p.a);
    }
};


template <sm::concepts::Number T> class formatter<sm::BoundingBox<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::BoundingBox<T>& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(),
                              R"(
BoundingBox(min = {},
            max = {}))",
                              p.min, p.max);
    }
};

template <> class formatter<sm::LineSegment>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::LineSegment& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "LineSegment({}, {})", p.p1, p.p2);
    }
};

template <> class formatter<sm::Transform>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::Transform& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "Transform[pos: {}, scale: {}]", p.pos, p.scale);
    }
};

template <> class formatter<std::complex<sm::f64>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const std::complex<sm::f64>& p, auto& ctx)
    {
        return fmt::format_to(ctx.out(), "complex({:.4}, {:.4})", p.real(), p.imag());
    }
};
} // namespace fmt

namespace sm
{
inline auto date_time_str()
{
    const auto now      = std::chrono::system_clock::now();
    const auto duration = now.time_since_epoch();
    const auto millis   = std::chrono::duration_cast<std::chrono::milliseconds>(duration) -
                        std::chrono::duration_cast<std::chrono::seconds>(duration);
    return fmt::format("{:%Y-%m-%d_%H-%M-%S-}{:03}", now, millis.count());
}
} // namespace sm
