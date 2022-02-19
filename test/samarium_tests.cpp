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

#include "samarium/samarium.hpp"

using sm::util::print;

int main()
{
    using namespace sm::literals;
    auto rn = sm::Renderer{sm::Image{sm::dimsHD}};

    const auto gravity = -100.0_y;

    const auto anchor          = 30.0_y;
    const auto rest_length     = 4.0;
    const auto spring_constant = 10.0;

    auto p1 = sm::Particle{.pos = {}, .vel = {50, 0}, .radius = 5, .mass = 4};
    auto p2 = p1;

    const auto l = sm::LineSegment{{-30, -30}, {30, -9}};

    const auto dims         = rn.image.dims.as<double>();
    const auto viewport_box = rn.viewport_box();

    auto win = sm::Window{rn.image.dims, "Collision", 60};
    while (win.is_open() && win.frame_counter <= 1800 && 1)
    {
        sm::WindowManager wm(win, rn, "#10101B"_c);

        p1.apply_force(p1.mass * gravity);
        const auto spring = p1.pos - anchor;
        const auto force =
            spring.with_length(spring_constant * (rest_length - spring.length()));
        print(spring, force);
        p1.apply_force(force);
        p1.update(1.0 / 60.0);
        // sm::phys::collide(p1, p2, l);

        for (auto&& i : viewport_box) sm::phys::collide(p1, p2, i);

        // rn.draw(l, "#03bcff"_c);
        rn.draw(sm::LineSegment{anchor, p1.pos}, sm::colors::lightgreen, .08);
        rn.draw(p1, sm::colors::orangered);
        p2 = p1;
    }
}
