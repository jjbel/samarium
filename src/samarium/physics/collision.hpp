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

#include "samarium/math/geometry.hpp"

#include "Particle.hpp"

namespace sm::phys
{
constexpr std::optional<Vector2> did_collide(Particle p1, Particle p2)
{
    if (math::distance(p1.pos, p2.pos) <= p1.radius + p2.radius)
        return std::optional((p1.pos + p2.pos) / 2.0);
    else
        return std::nullopt;
}

constexpr auto collide(Particle& p1, Particle& p2)
{
    // https://courses.lumenlearning.com/boundless-physics/chapter/collisions/#:~:text=particles%20are%20involved%20in%20an-,elastic%20collision,-%2C%20the%20velocity%20of%20the%20first
    if (auto point = did_collide(p1, p2))
    {
        const auto shift = (p1.radius + (math::distance(p1.pos, p2.pos) - p2.radius)) / 2;
        fmt::print("Shift: {}\n", shift);
        const auto centre = p1.pos + (p2.pos - p1.pos).with_length(shift);
        p1.pos            = centre + (p1.pos - centre).with_length(p1.radius);
        p2.pos            = centre + (p2.pos - centre).with_length(p2.radius);


        const auto vel1 = (p1.mass - p2.mass) / (p2.mass + p1.mass) * p1.vel +
                          2 * p2.mass / (p2.mass + p1.mass) * p2.vel;
        const auto vel2 = 2 * p1.mass / (p2.mass + p1.mass) * p1.vel +
                          (p2.mass - p1.mass) / (p2.mass + p1.mass) * p2.vel;
        p1.vel = vel1;
        p2.vel = vel2;
    }
}

constexpr auto did_collide(Particle now, Particle prev, LineSegment l)
{
    return sm::math::clamped_intersection({prev.pos, now.pos}, l);
}
} // namespace sm::phys
