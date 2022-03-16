/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../core/ThreadPool.hpp"
#include "../math/Transform.hpp"
#include "../math/geometry.hpp"
#include "../physics/Particle.hpp"

#include "Gradient.hpp"
#include "Image.hpp"
#include "Trail.hpp"

namespace sm
{
namespace concepts
{
// takes a const Vector2& and returns a Color
template <typename T>
concept DrawableLambda = std::is_invocable_r_v<Color, T, const Vector2&>;
} // namespace concepts

class Renderer
{
  public:
    static auto rasterize(f64 distance, f64 radius, f64 aa_factor)
    {
        // https://www.desmos.com/calculator/jhewyqc2wy
        return interp::clamp((radius - distance) / aa_factor + 1, {0.0, 1.0});
    }

    static auto rasterize(Color color, f64 distance, f64 radius, f64 aa_factor)
    {
        return color.with_multiplied_alpha(rasterize(distance, radius, aa_factor));
    }

    // -----------------MEMBERS---------------//
    Image image;
    Transform transform{.pos   = image.dims.as<f64>() / 2.,
                        .scale = Vector2{10, 10} * Vector2{1.0, -1.0}};

    Renderer(const Image& image_, u32 thread_count_ = std::thread::hardware_concurrency())
        : image{image_}, thread_count{thread_count_}, thread_pool{thread_count_}

    {
    }

    void fill(const Color& color) { this->image.fill(color); }

    template <concepts::DrawableLambda T> void draw(T&& fn)
    {
        const auto rect = image.rect();
        this->draw(std::forward<T>(fn),
                   transform.apply_inverse(
                       Rect{.min = rect.min, .max = rect.max + Indices{1, 1}}.template as<f64>()));
    }

    void draw(concepts::DrawableLambda auto&& fn, const Rect<f64>& rect)
    {
        const auto box = this->transform.apply(rect)
                             .clamped_to(image.rect().template as<f64>())
                             .template as<size_t>();

        if (math::area(box) == 0UL) return;

        for (size_t y = box.min.y; y < box.max.y; y++)
        {
            for (size_t x = box.min.x; x < box.max.x; x++)
            {
                const auto coords             = Indices{x, y};
                const auto coords_transformed = transform.apply_inverse(coords.template as<f64>());

                // image[coords].add_alpha_over(Color{255, 80, 200, 100});
                const auto col = fn(coords_transformed);

                image[coords].add_alpha_over(col);
            }
        }
    }

    void draw(Circle circle, Color color, f64 aa_factor = 1.6);

    void draw(const Particle& particle, f64 aa_factor = 0.1);

    void draw(const Particle& particle, Color color, f64 aa_factor = 0.3);

    void draw(const Trail& trail,
              Color color     = Color{100, 255, 80},
              f64 fade_factor = 1.0,
              f64 radius      = 0.2,
              f64 aa_factor   = 0.1);

    void draw(const Trail& trail,
              concepts::Interpolator auto&& fn,
              f64 radius    = 1.0,
              f64 aa_factor = 0.1)
    {
        const auto mapper =
            interp::make_mapper<f64>({0.0, static_cast<double>(trail.size())}, {0.0, 1.0});

        for (double i = 0.0; const auto& pos : trail.span())
        {
            this->draw(Circle{pos, radius}, fn(mapper(i)), aa_factor);
            i += 1.0;
        }
    }

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1,
                           f64 aa_factor = 0.1);

    void draw_line(const LineSegment& ls,
                   Color color   = Color{255, 255, 255},
                   f64 thickness = 0.1,
                   f64 aa_factor = 0.1);

    void draw_line_segment(const LineSegment& ls,
                           concepts::Interpolator auto&& function_along_line,
                           f64 thickness = 0.1,
                           f64 aa_factor = 0.1)
    {
        const auto vector = ls.vector().abs();
        const auto extra  = 2 * (aa_factor + thickness);
        this->draw(
            [&function_along_line, &ls, thickness, aa_factor](const Vector2& coords)
            {
                return rasterize(function_along_line(math::lerp_along(coords, ls)),
                                 math::clamped_distance(coords, ls), thickness, aa_factor);
            },
            Rect<f64>::from_centre_width_height((ls.p1 + ls.p2) / 2.0, vector.x + extra,
                                                vector.y + extra));
    }

    void draw_line(const LineSegment& ls,
                   concepts::Interpolator auto&& function_along_line,
                   f64 thickness = 0.1,
                   f64 aa_factor = 2)
    {
        const auto vector = ls.vector().abs();
        const auto extra  = 2 * aa_factor;
        this->draw(
            [&function_along_line, &ls, thickness, aa_factor](const Vector2& coords)
            {
                return rasterize(function_along_line(math::clamped_lerp_along(coords, ls)),
                                 math::distance(coords, ls), thickness, aa_factor);
            });
    }

    void draw_grid(bool axes = true, bool grid = true, bool dots = true);

    void render() { this->thread_pool.wait_for_tasks(); }

    std::array<LineSegment, 4> viewport_box() const;

  private:
    u32 thread_count;
    sm::ThreadPool thread_pool;
};
} // namespace sm
