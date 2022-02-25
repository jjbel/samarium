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

#include <execution>
#include <utility>

#include "Renderer.hpp"

namespace sm
{
void Renderer::render()
{
    if (draw_funcs.empty()) return;

    const auto size = image.size();
    size_t j        = 0;

    for (size_t i = 0; i < thread_count; i++)
    {
        const auto chunk_size = i < size % thread_count ? size / thread_count + 1
                                                        : size / thread_count;

        const auto task = [chunk_size, j, dims = image.dims, &image = this->image,
                           &draw_funcs = this->draw_funcs, tr = this->transform]
        {
            for (size_t k = j; k < j + chunk_size; k++)
            {
                const auto coords =
                    tr.apply_inverse(sm::convert_1d_to_2d(dims, k).as<double>());
                for (const auto& drawer : draw_funcs)
                {
                    if (drawer.rect.contains(coords))
                    {
                        image[k].add_alpha_over(drawer.fn(coords));
                    }
                }
            }
        };

        thread_pool.push_task(task);
        j += chunk_size;
    }
    thread_pool.wait_for_tasks();
    draw_funcs.clear();
}

void Renderer::draw(Circle circle, Color color, double_t aa_factor)
{
    this->draw(
        [=](const Vector2& coords)
        {
            return antialias(color, math::distance(coords, circle.centre),
                             circle.radius, aa_factor);
        },
        Rect<double_t>::from_centre_width_height(
            circle.centre, circle.radius + aa_factor, circle.radius + aa_factor));
}

void Renderer::draw(const Particle& particle, double_t aa_factor)
{
    this->draw(particle.as_circle(), particle.color, aa_factor);
}

void Renderer::draw(const Particle& particle, Color color, double_t aa_factor)
{
    this->draw(particle.as_circle(), color, aa_factor);
}

void Renderer::draw_line_segment(const LineSegment& ls,
                                 Color color,
                                 double_t thickness,
                                 double_t aa_factor)
{
    const auto vector = ls.vector().abs();
    const auto extra  = 2 * aa_factor;
    this->draw(
        [=](const Vector2& coords)
        {
            return antialias(color, math::clamped_distance(coords, ls), thickness,
                             aa_factor);
        },
        Rect<double_t>::from_centre_width_height(
            (ls.p1 + ls.p2) / 2.0, vector.x + extra, vector.y + extra));
}

void Renderer::draw_line(const LineSegment& ls,
                         Color color,
                         double_t thickness,
                         double_t aa_factor)
{
    this->draw(
        [=](const Vector2& coords) {
            return antialias(color, math::distance(coords, ls), thickness,
                             aa_factor);
        });
}

void Renderer::draw_grid(bool axes, bool grid, bool dots)
{
    // if (axes)
    // {
    //     this->draw_line_segment(
    //         transform.apply_inverse(LineSegment{{image.dims.x / 2.0},
    //         {image.dims.x / 2.0, image.dims.y}}), Color{255, 255, 255, 0});
    // }
}

std::array<LineSegment, 4> Renderer::viewport_box() const
{
    const auto double_dims = this->image.dims.as<double>();
    return std::array{
        this->transform.apply_inverse(sm::LineSegment{{}, {0, double_dims.y}}),
        this->transform.apply_inverse(sm::LineSegment{{}, {double_dims.x, 0}}),
        this->transform.apply_inverse(
            sm::LineSegment{{double_dims.x, 0}, {double_dims.x, double_dims.y}}),
        this->transform.apply_inverse(
            sm::LineSegment{{0, double_dims.y}, {double_dims.x, double_dims.y}})};
}
} // namespace sm
