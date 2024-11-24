/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "fmt/chrono.h"
#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

#include "samarium/core/version.hpp"
#include "samarium/graphics/Color.hpp"
#include "samarium/math/BoundingBox.hpp"
#include "samarium/math/Transform.hpp"
#include "samarium/math/Vec2.hpp"
#include "samarium/math/complex.hpp"
#include "samarium/math/shapes.hpp"
#include "samarium/physics/Particle.hpp"
#include "samarium/util/SourceLocation.hpp"

namespace fmt
{
template <sm::concepts::Integral T> class formatter<sm::Vec2_t<T>>
{
  public:
    // as of fmt 11, should be marked const
    // https://fmt.dev/11.0/api/
    // TODO parse shouldn't be const?

    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    constexpr auto format(sm::Vec2_t<T> p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "({:>3}, {:>3})", p.x, p.y);
    }
};

template <sm::concepts::FloatingPoint T> class formatter<sm::Vec2_t<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    constexpr auto format(sm::Vec2_t<T> p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "({:6.3f}, {:6.3f})", p.x, p.y);
    }
};

template <typename T, typename Char> struct is_tuple_formattable<sm::Vec2_t<T>, Char>
{
    static constexpr auto value = false;
};

template <> class formatter<sm::Version>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::Version& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "samarium version {}.{}.{}", p.major, p.minor, p.patch);
    }
};

template <typename Float> class formatter<sm::Particle<Float>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::Particle<Float>& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "Particle(pos: {}, vel: {}, acc: {})", p.pos, p.vel,
                              p.acc);
    }
};

template <> class formatter<sm::Color>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::Color& p, auto& ctx) const
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

    auto format(const sm::BoundingBox<T>& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(),
                              R"(
BoundingBox(min = {},
            max = {})
)",
                              p.min, p.max);
    }
};

template <> class formatter<sm::LineSegment>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::LineSegment& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "LineSegment({}, {})", p.p1, p.p2);
    }
};

template <> class formatter<sm::Transform>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::Transform& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "[pos: {}, scale: {}]", p.pos, p.scale);
    }
};

template <> class formatter<std::complex<sm::f64>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const std::complex<sm::f64>& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "complex({:.4}, {:.4})", p.real(), p.imag());
    }
};

template <> class formatter<sm::SourceLocation>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    auto format(const sm::SourceLocation& p, auto& ctx) const
    {
        return fmt::format_to(ctx.out(), "[{}:{}:{} in {}()]", p.file_name(), p.line(), p.column(),
                              p.function_name());
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
