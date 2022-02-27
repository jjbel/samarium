/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <functional>
#include <mutex>
#include <ranges>

#include "samarium/core/ThreadPool.hpp"
#include "samarium/math/Transform.hpp"
#include "samarium/math/geometry.hpp"
#include "samarium/physics/Particle.hpp"

#include "Gradient.hpp"
#include "Image.hpp"
#include "Trail.hpp"
#include "colors.hpp"

namespace sm
{
class Renderer
{
  public:
    static auto antialias(double_t distance, double_t radius, double_t aa_factor)
    {
        // https://www.desmos.com/calculator/jhewyqc2wy
        return interp::clamp((radius - distance) / aa_factor + 1, {0.0, 1.0});
    }

    static auto
    antialias(Color color, double_t distance, double_t radius, double_t aa_factor)
    {
        return color.with_multiplied_alpha(
            antialias(distance, radius, aa_factor));
    }

    struct Drawer
    {
        std::function<sm::Color(const sm::Vector2&)> fn;
        sm::Rect<double_t> rect;
    };


    Image image;
    Transform transform{.pos   = image.dims.as<double_t>() / 2.,
                        .scale = Vector2{10, 10} * Vector2{1.0, -1.0}};

    Renderer(const Image& image_,
             u32 thread_count_ = std::thread::hardware_concurrency())
        : image{image_}, thread_count{thread_count_}, thread_pool{thread_count_}

    {
    }

    auto fill(const Color& color) { this->image.fill(color); }

    template <typename T>
    void draw(T&& fn) requires(
        concepts::reason("Function should accept a const Vector2&") &&
        std::invocable<T, const Vector2&>)
    {
        const auto rect = image.rect();
        this->draw_funcs.emplace_back(
            Drawer(fn, transform.apply_inverse(
                           Rect{.min = rect.min, .max = rect.max + Indices{1, 1}}
                               .template as<double_t>())));
    }

    template <typename T>
    void draw(T&& fn, Rect<double_t> rect) requires(
        concepts::reason("Function should accept a const Vector2&") &&
        std::invocable<T, const Vector2&>)
    {
        this->draw_funcs.emplace_back(Drawer{fn, rect});
    }

    void draw(Circle circle, Color color, double_t aa_factor = 1.6);

    void draw(const Particle& particle, double_t aa_factor = 0.1);

    void draw(const Particle& particle, Color color, double_t aa_factor = 0.1);

    void draw(const Trail& trail,
              Color color          = sm::colors::lightgreen,
              double_t fade_factor = 0.0,
              double_t radius      = 1.0,
              double_t aa_factor   = 0.1)
    {
    }

    template <size_t gradient_size>
    void draw(const Trail& trail,
              Gradient<gradient_size> color,
              double_t fade_factor = 0.0,
              double_t radius      = 1.0,
              double_t aa_factor   = 0.1)
    {
    }

    void draw_line_segment(const LineSegment& ls,
                           Color color        = sm::colors::white,
                           double_t thickness = 0.1,
                           double_t aa_factor = 0.1);

    void draw_line(const LineSegment& ls,
                   Color color        = sm::colors::white,
                   double_t thickness = 0.1,
                   double_t aa_factor = 0.1);

    template <typename T>
    requires std::invocable<T, double>
    void draw_line_segment(const LineSegment& ls,
                           T function_along_line,
                           double_t thickness = 0.1,
                           double_t aa_factor = 0.1)
    {
        const auto vector = ls.vector().abs();
        const auto extra  = 2 * aa_factor;
        this->draw(
            [=](const Vector2& coords)
            {
                return antialias(
                    function_along_line(math::lerp_along(coords, ls)),
                    math::clamped_distance(coords, ls), thickness, aa_factor);
            },
            Rect<double_t>::from_centre_width_height(
                (ls.p1 + ls.p2) / 2.0, vector.x + extra, vector.y + extra));
    }

    template <typename T>
    requires std::invocable<T, double>
    void draw_line(const LineSegment& ls,
                   T function_along_line,
                   double_t thickness = 0.1,
                   double_t aa_factor = 0.1)
    {
        const auto vector = ls.vector().abs();
        const auto extra  = 2 * aa_factor;
        this->draw(
            [=](const Vector2& coords)
            {
                return antialias(
                    function_along_line(math::clamped_lerp_along(coords, ls)),
                    math::distance(coords, ls), thickness, aa_factor);
            });
    }

    void draw_grid(bool axes = true, bool grid = true, bool dots = true);

    void render();

    std::array<LineSegment, 4> viewport_box() const;


  private:
    u32 thread_count;
    sm::ThreadPool thread_pool;
    std::vector<Drawer> draw_funcs{};
};
} // namespace sm
