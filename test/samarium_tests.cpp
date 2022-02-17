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
#include <functional>
#include <ranges>

#include "samarium/samarium.hpp"

using sm::util::print;
namespace si = sm::interp;

int main()
{
    // using namespace sm::literals;
    // auto rn = sm::Renderer{sm::Image{sm::dimsHD}};

    // const auto gravity = -50.0_y;

    // auto p1 = sm::Particle{.pos = {-36, 0}, .vel = {6, 30}, .radius = 4};
    // auto p2 = p1;

    // auto l = sm::LineSegment{{-30, -30}, {30, -29}};

    // const auto dims = rn.image.dims;
    // const auto l1   = sm::LineSegment{{-10.0, 10.0}, {10.0, 10.0}};
    // const auto l2   = sm::LineSegment{{-10.0, 10.0}, {-11.0, -10.0}};
    // const auto l3   = sm::LineSegment{{-10.0, -10.0}, {10.0, -10.0}};
    // const auto l4   = sm::LineSegment{{10.0, 10.0}, {10.0, -10.0}};

    // auto win = sm::Window{rn.image.dims, "Collision", 60};
    // sm::util::Stopwatch w{};

    // // print(sm::math::clamped_distance(sm::Vector2{1, 1}, l2));

    // while (win.is_open() && win.frame_counter <= 1200 && 1)
    // {
    //     sm::WindowManager(win, rn, "#10101B"_c);

    //     p1.acc = gravity;
    //     p1.update(1.0 / 60.0);
    //     sm::phys::collide(p1, p2, l);

    //     rn.draw(l, "#03bcff"_c);
    //     // rn.draw(l1, "#03bcff"_c);
    //     // rn.draw(l2, "#03bcff"_c);
    //     // rn.draw(l3, "#03bcff"_c);
    //     // rn.draw(l4, "#03bcff"_c);

    //     // p1.pos = l2.p2;
    //     rn.draw(p1, sm::colors::orangered);
    //     p2 = p1;
    // }
    // w.print();
}
