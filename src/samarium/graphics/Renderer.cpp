/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "Renderer.hpp"

namespace sm
{
void Renderer::draw(Circle circle, Color color, f64 aa_factor)
{
    this->draw(
        [circle, color, aa_factor](const Vector2& coords)
        {
            return rasterize(color, math::distance(coords, circle.centre),
                             circle.radius, aa_factor);
        },
        Rect<f64>::from_centre_width_height(
            circle.centre, circle.radius + aa_factor, circle.radius + aa_factor));
}

void Renderer::draw(const Particle& particle, f64 aa_factor)
{
    this->draw(particle.as_circle(), particle.color, aa_factor);
}

void Renderer::draw(const Particle& particle, Color color, f64 aa_factor)
{
    this->draw(particle.as_circle(), color, aa_factor);
}

void Renderer::draw_line_segment(const LineSegment& ls,
                                 Color color,
                                 f64 thickness,
                                 f64 aa_factor)
{
    const auto vector = ls.vector().abs();
    const auto extra  = 2 * aa_factor;
    this->draw(
        [&ls, color, thickness, aa_factor](const Vector2& coords)
        {
            return rasterize(color, math::clamped_distance(coords, ls), thickness,
                             aa_factor);
        },
        Rect<f64>::from_centre_width_height((ls.p1 + ls.p2) / 2.0,
                                            vector.x + extra, vector.y + extra));
}

void Renderer::draw_line(const LineSegment& ls,
                         Color color,
                         f64 thickness,
                         f64 aa_factor)
{
    this->draw(
        [color, &ls, thickness, aa_factor](const Vector2& coords) {
            return rasterize(color, math::distance(coords, ls), thickness,
                             aa_factor);
        });
}

// void Renderer::draw_grid(bool axes, bool grid, bool dots)
// {
// if (axes)
// {
//     this->draw_line_segment(
//         transform.apply_inverse(LineSegment{{image.dims.x / 2.0},
//         {image.dims.x / 2.0, image.dims.y}}), Color{255, 255, 255, 0});
// }
// }

std::array<LineSegment, 4> Renderer::viewport_box() const
{
    const auto double_dims = this->image.dims.as<f64>();
    return std::array{
        this->transform.apply_inverse(sm::LineSegment{{}, {0, double_dims.y}}),
        this->transform.apply_inverse(sm::LineSegment{{}, {double_dims.x, 0}}),
        this->transform.apply_inverse(
            sm::LineSegment{{double_dims.x, 0}, {double_dims.x, double_dims.y}}),
        this->transform.apply_inverse(
            sm::LineSegment{{0, double_dims.y}, {double_dims.x, double_dims.y}})};
}
} // namespace sm
