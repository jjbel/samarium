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

#include "samarium/Colors.hpp"
#include "samarium/Renderer.hpp"
#include "samarium/ThreadPool.hpp"
#include "samarium/Window.hpp"
#include "samarium/file.hpp"
#include "samarium/interp.hpp"
#include <execution>
#include <functional>
#include <ranges>


using sm::util::print;
namespace si = sm::interp;

int main()
{
    using namespace sm::literals;
    auto rn = sm::Renderer{ sm::Image{ sm::dimsHD, "#10101B"_c } };

    for (size_t i = 0; i < 10; i++)
    {
        rn.draw(sm::Circle{ .centre = (.8_x + .6_y) * (30. * i) + (100.0_x + 100.0_y),
                            .radius = 24. },
                sm::colors::red, 2.);
    }


    auto w = sm::util::Stopwatch{};

    rn.render();

    w.print();
    w.reset();
    // sm::file::export_to(rn.image, "temp1.tga", true);
    w.print();
    print(rn.transform);
    // for (auto win = sm::Window{ rn.image.dims }; win;)
    // {
    //     win.get_input();

    //     win.draw(rn.image);
    //     win.display();
    // }
}
