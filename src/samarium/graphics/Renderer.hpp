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
concept DrawableLambda = std::is_invocable_r<Color, T, const Vector2&>::value;
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
    std::mutex m;
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

        if (math::area(box) == 0ul) return;

        const auto height = box.height();

        const auto current_thread_count = std::min(static_cast<size_t>(this->thread_count), height);

        auto j = size_t{};

        for (auto i = size_t{}; i < current_thread_count; i++)
        {
            const auto chunk_size = i < height % current_thread_count
                                        ? height / current_thread_count + 1
                                        : height / current_thread_count;

            const auto task =
                [chunk_size, j, box, fn, tr = this->transform, &image = this->image, &m = this->m]
            {
                // std::scoped_lock lock1{m};
                Extents{j + box.min.y, j + box.min.y + chunk_size - 1}.for_each(
                    [box, tr, &image, &fn, &m](auto y)
                    {
                        Extents{box.min.x, box.max.x}.for_each(
                            [y, tr, &image, &fn, &m](auto x)
                            {
                                const auto coords = Indices{x, y};
                                const auto coords_transformed =
                                    tr.apply_inverse(coords.template as<f64>());

                                // image[coords].add_alpha_over(
                                //     Color{255, 80, 200, 100});
                                const auto col = fn(coords_transformed);

                                image[coords].add_alpha_over(col);
                            });
                    });
            };
            this->thread_pool.push_task(task);
            j += chunk_size;
        }
        // this->render();
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
