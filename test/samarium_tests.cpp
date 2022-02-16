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
    auto rn = sm::Renderer{sm::Image{sm::dimsHD, "#10101B"_c}};

    const auto bg   = "#10101B"_c;
    auto ball       = sm::Particle{.vel{1, 1}, .radius = 40, .color = "#ff3721"_c};
    const auto rect = sm::Rect<double>{{-40, -40}, 40, 40};
    print(rn.transform.apply(rect).min);
    print(rn.transform.apply(rect).max);

    auto win = sm::Window{rn.image.dims};
    sm::util::Stopwatch w{};
    while (win.is_open() && win.frame_counter <= 600 && 1)
    {
        win.get_input();
        rn.fill(bg);

        ball.update();
        rn.draw(ball);
        rn.draw(sm::LineSegment{{0,0}, {500, 800}}, "#0dbaff"_c, 10);

        rn.render();

        // sm::file::export_to(rn.image, fmt::format("dev/temp{:4}.tga",
        // win.frame_counter), true);

        win.draw(rn.image);
        win.display();
    }
    auto t = w.time().count();
    fmt::print("Frames: {}, time: {:.3}, framerate: {:.3} fps, time per frame: {:.3}ms\n",
               win.frame_counter, t, win.frame_counter / t, t / win.frame_counter * 1000);

    sm::file::export_to(rn.image);
}
