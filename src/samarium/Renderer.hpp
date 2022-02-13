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

#include <functional>
#include <mutex>
#include <ranges>
#include <thread>
#include <vector>

#include "Image.hpp"
#include "ThreadPool.hpp"
#include "Transform.hpp"
#include "interp.hpp"
#include "shapes.hpp"

namespace sm
{
class Renderer
{
  public:
    struct Drawer
    {
        std::function<sm::Color(const sm::Vector2&)> fn;
        sm::Rect<double> rect;
    };


    Image image;
    Transform transform{ image.dims.as<double>() / 2., 1. };

    Renderer(const Image& image_, u32 thread_count_ = std::thread::hardware_concurrency())
        : image{ image_ }, thread_count{ thread_count_ }, thread_pool{ thread_count_ }

    {
    }

    void draw(Drawer&& drawer);

    void draw(Circle circle, Color color, double aa_factor = 2.);

    void render();


  private:
    u32 thread_count;
    sm::ThreadPool thread_pool;
    std::vector<Drawer> draw_funcs{};
};
} // namespace sm
