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

int main()
{
    using namespace sm::literals;
    auto rn = sm::Renderer{sm::Image{sm::dimsHD}};

    const auto gravity = -100.0_y;

    const sm::Vector2 anchor   = 30.0_y;
    const auto rest_length     = 14.0;
    const auto spring_constant = 100.0;

    auto p1 = sm::Particle{.pos = {}, .vel = {50, 0}, .radius = 3, .mass = 40};
    auto p2 = p1;

    const auto l = sm::LineSegment{{-30, -30}, {30, -9}};

    const auto dims         = rn.image.dims.as<double>();
    const auto viewport_box = rn.viewport_box();

    auto window = sm::Window{rn.image.dims, "Collision", 60};

    sm::util::Stopwatch watch{};

    const auto run_every_frame = [&]
    {
        p1.apply_force(p1.mass * gravity);
        const auto spring = p1.pos - anchor;
        const auto force =
            spring.with_length(spring_constant * (rest_length - spring.length()));
        p1.apply_force(force);
        p1.update();

        sm::phys::collide(p1, p2, l);

        for (auto&& i : viewport_box) sm::phys::collide(p1, p2, i);
        rn.draw_line_segment(l, sm::gradients::purple, 0.4);
        rn.draw_line_segment(sm::LineSegment{anchor, p1.pos}, "#c471ed"_c, .06);
        rn.draw(p1, sm::colors::red);
        p2 = p1;

        fmt::print(stderr, "\r{:>{}}", "",
                   sm::util::get_terminal_dims()
                       .x); // clear line by padding spaces to width of terminal
        fmt::print(
            stderr, "\rCurrent framerate: {}",
            std::round(
                1.0 /
                watch.time().count())); // print to stderr for no line buffering
        watch.reset();
    };

    window.run(rn, "#10101B"_c, run_every_frame);
}
