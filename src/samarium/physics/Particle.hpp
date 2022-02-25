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

#include "samarium/graphics/Color.hpp"
#include "samarium/math/shapes.hpp"

namespace sm
{
struct Particle
{
    Vector2 pos{};
    Vector2 vel{};
    Vector2 acc{};
    double_t radius{1};
    double_t mass{1};
    Color color{};

    constexpr auto as_circle() const noexcept { return Circle{pos, radius}; }

    constexpr auto apply_force(Vector2 force) noexcept { acc += force / mass; }

    constexpr auto update(double_t time_delta = 1.0 / 64) noexcept
    {
        vel += acc * time_delta;
        pos += vel * time_delta;
        acc *= 0.0;
    }
};

constexpr auto update(auto& object, double_t time_delta = 1.0 / 60)
{
    object.update(time_delta);
}
} // namespace sm
