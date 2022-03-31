/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>   // for array
#include <cstddef> // for size_t
#include <span>    // for span
#include <thread>  // for thread
#include <utility> // for forward

#include "../core/ThreadPool.hpp"  // for ThreadPool
#include "../core/types.hpp"       // for f64, u64, u32
#include "../graphics/Color.hpp"   // for Color
#include "../math/BoundingBox.hpp" // for BoundingBox
#include "../math/Extents.hpp"     // for Extents, Extents<>::Iterator
#include "../math/Transform.hpp"   // for Transform
#include "../math/Vector2.hpp"     // for Vector2_t, Vector2, operator+
#include "../math/geometry.hpp"    // for area, clamped_distance, cla...
#include "../math/interp.hpp"      // for clamp, make_mapper
#include "../math/shapes.hpp"      // for LineSegment, Circle
#include "../physics/Particle.hpp" // for Particle

#include "Image.hpp" // for Image, dimsFHD
#include "Trail.hpp" // for Trail

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

    Renderer(const Image& image_ = sm::Image{sm::dimsFHD},
             u32 thread_count_   = std::thread::hardware_concurrency())
        : image{image_}, thread_count{thread_count_}, thread_pool{thread_count_}

    {
    }

    void fill(const Color& color) { this->image.fill(color); }

    template <concepts::DrawableLambda T> void draw(T&& fn)
    {
        const auto bounding_box = image.bounding_box();
        this->draw(std::forward<T>(fn), transform.apply_inverse(BoundingBox<u64>{
                                            .min = bounding_box.min,
                                            .max = bounding_box.max +
                                                   Indices{1, 1}}.template as<f64>()));
    }

    void draw(concepts::DrawableLambda auto&& fn, const BoundingBox<f64>& bounding_box)
    {
        const auto box = this->transform.apply(bounding_box)
                             .clamped_to(image.bounding_box().template as<f64>())
                             .template as<size_t>();

        if (math::area(box) == 0UL) return;

        for (auto y : box.y_range())
        {
            for (auto x : box.x_range())
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
            BoundingBox<f64>::from_centre_width_height((ls.p1 + ls.p2) / 2.0, vector.x + extra,
                                                       vector.y + extra));
    }

    void draw_line(const LineSegment& ls,
                   concepts::Interpolator auto&& function_along_line,
                   f64 thickness = 0.1,
                   f64 aa_factor = 2)
    {
        // const auto vector = ls.vector().abs();
        // const auto extra  = 2 * aa_factor;
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
