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
    using namespace sm::literals;
    auto rn = sm::Renderer{sm::Image{sm::dimsHD}};

    auto ball_now  = sm::Particle{.vel{30, -7}, .radius = 40, .color = "#ff3721"_c};
    auto ball_prev = ball_now;

    auto l = sm::LineSegment{{-300, -300}, {300, -270}};

    auto win = sm::Window{rn.image.dims};
    sm::util::Stopwatch w{};
    while (win.is_open() && win.frame_counter <= 600 && 1)
    {
        sm::WindowManager(win, rn, "#10101B"_c);

        ball_prev = ball_now;
        ball_now.update();

        rn.draw(l, "#0dbaff"_c);
        rn.draw(ball_now);

        auto point = sm::phys::did_collide(ball_now, ball_prev, l);
        if (point)
        {
            print("Point: ", *point);
            rn.draw(sm::Circle{*point, 4}, sm::colors::green);
        }
    }
}
