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

    const auto gravity = -50.0_y;

    auto p1 = sm::Particle{.pos = {-36, 0}, .vel = {18, 0}, .radius = 1};
    // auto p2 =
    // sm::Particle{.pos = {0, -200}, .vel = {0, 100}, .radius = 40, .color = sm::colors::red};
    auto l = sm::LineSegment{{-30, -30}, {30, -29}};

    auto win = sm::Window{rn.image.dims};
    sm::util::Stopwatch w{};

    while (win.is_open() && win.frame_counter <= 800 && 1)
    {
        sm::WindowManager(win, rn, "#10101B"_c);

        p1.acc = gravity;
        // p2.acc = gravity;

        // sm::phys::collide(p1, p2);
        p1.update(1.0 / 60.0);
        // p2.update(1.0 / 60.0);
        rn.draw(l);
        rn.draw(p1, sm::colors::orangered);
        // rn.draw(p2);
    }
    w.print();
}
