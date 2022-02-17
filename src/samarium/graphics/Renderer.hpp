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

#include <functional>
#include <mutex>
#include <ranges>
#include <thread>
#include <vector>

#include "samarium/core/ThreadPool.hpp"
#include "samarium/math/Transform.hpp"
#include "samarium/graphics/colors.hpp"
#include "samarium/math/geometry.hpp"
#include "samarium/physics/Particle.hpp"

#include "Image.hpp"

namespace sm
{
class Renderer
{
  public:
    struct Drawer
    {
        std::function<sm::Color(const sm::Vector2&)> fn;
        sm::Rect<double> rect;
    };


    Image image;
    Transform transform{.pos   = image.dims.as<double>() / 2.,
                        .scale = Vector2{10, 10} * Vector2{1.0, -1.0}};

    Renderer(const Image& image_, u32 thread_count_ = std::thread::hardware_concurrency())
        : image{image_}, thread_count{thread_count_}, thread_pool{thread_count_}

    {
    }

    auto fill(const Color& color) { this->image.fill(color); }

    template <typename T>
    void draw(T&& fn) requires(concepts::reason("Function should accept a const Vector2&") &&
                               std::invocable<T, const Vector2&>)
    {
        const auto rect = image.rect();
        this->draw_funcs.emplace_back(Drawer(
            fn,
            transform.apply_inverse(
                Rect{.min = rect.min, .max = rect.max + Indices{1, 1}}.template as<double>())));
    }

    template <typename T>
    void draw(T&& fn, Rect<double> rect) requires(
        concepts::reason("Function should accept a const Vector2&") &&
        std::invocable<T, const Vector2&>)
    {
        this->draw_funcs.emplace_back(Drawer{fn, rect});
    }

    void draw(Circle circle, Color color, double aa_factor = 1.6);

    void draw(Particle particler, Color color = sm::colors::orangered, double aa_factor = 0.1);

    void draw(LineSegment ls,
              Color color      = sm::colors::white,
              double thickness = 0.1,
              double aa_factor = 0.1);

    void draw_grid(bool axes = true, bool grid = true, bool dots = true);

    void render();


  private:
    u32 thread_count;
    sm::ThreadPool thread_pool;
    std::vector<Drawer> draw_funcs{};
};
} // namespace sm
