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
template <typename T>
concept DrawableLambda =
    std::is_invocable_r<Color, T, const Vector2&>::value; // takes a const
                                                          // Vector2& and returns
                                                          // a Color
}

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
        return color.with_multiplied_alpha(
            rasterize(distance, radius, aa_factor));
    }

    Image image;
    Transform transform{.pos   = image.dims.as<f64>() / 2.,
                        .scale = Vector2{10, 10} * Vector2{1.0, -1.0}};

    Renderer(const Image& image_,
             u32 thread_count_ = std::thread::hardware_concurrency())
        : image{image_}, thread_count{thread_count_}, thread_pool{thread_count_}

    {
    }

    void fill(const Color& color) { this->image.fill(color); }

    void draw(concepts::DrawableLambda auto&& fn)
    {
        // const auto rect = image.rect();
        // this->draw_funcs.emplace_back(
        //     Drawer(fn, transform.apply_inverse(
        //                    Rect{.min = rect.min, .max = rect.max + Indices{1,
        //                    1}}
        //                        .template as<f64>())));
    }

    void draw(concepts::DrawableLambda auto&& fn, Rect<f64> rect)
    {
        // this->draw_funcs.emplace_back(Drawer{fn, rect});
        const auto box = this->transform.apply(rect)
                             .clamped_to(image.rect().template as<f64>())
                             .template as<size_t>();

        if (math::area(box) == 0) return;

        const auto task = [box, fn, dims = image.dims, tr = this->transform,
                           &image = this->image]
        {
            for (size_t y = box.min.y; y <= box.max.y; y++)
            {
                for (size_t x = box.min.x; x <= box.max.x; x++)
                {
                    const auto coords = Indices{x, y};
                    const auto coords_transformed =
                        tr.apply_inverse(coords.template as<f64>());
                    // image[coords].add_alpha_over(Color{80, 255, 80, 70});
                    if (box.contains(coords))
                    {
                        image[coords].add_alpha_over(fn(coords_transformed));
                    }
                }
            }
        };
        task();
        // this->thread_pool.push_task(task);
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
              f64 fade_factor = 1.0,
              f64 radius      = 1.0,
              f64 aa_factor   = 0.1);

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
        const auto extra  = 2 * aa_factor;
        this->draw(
            [&function_along_line, &ls, thickness,
             aa_factor](const Vector2& coords)
            {
                return rasterize(
                    function_along_line(math::lerp_along(coords, ls)),
                    math::clamped_distance(coords, ls), thickness, aa_factor);
            },
            Rect<f64>::from_centre_width_height(
                (ls.p1 + ls.p2) / 2.0, vector.x + extra, vector.y + extra));
    }

    void draw_line(const LineSegment& ls,
                   concepts::Interpolator auto&& function_along_line,
                   f64 thickness = 0.1,
                   f64 aa_factor = 2)
    {
        const auto vector = ls.vector().abs();
        const auto extra  = 2 * aa_factor;
        this->draw(
            [&function_along_line, &ls, thickness,
             aa_factor](const Vector2& coords)
            {
                return rasterize(
                    function_along_line(math::clamped_lerp_along(coords, ls)),
                    math::distance(coords, ls), thickness, aa_factor);
            });
    }

    void draw_grid(bool axes = true, bool grid = true, bool dots = true);

    std::array<LineSegment, 4> viewport_box() const;

  private:
    u32 thread_count;
    sm::ThreadPool thread_pool;
};
} // namespace sm
