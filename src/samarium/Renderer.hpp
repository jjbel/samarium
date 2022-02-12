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

// #include "boost/asio/thread_pool.hpp"
// #include "range/v3/range/conversion.hpp"
// #include "range/v3/view/chunk.hpp"
// #include "range/v3/view/transform.hpp"

#include "Image.hpp"
#include "Shapes.hpp"
#include "Transform.hpp"
#include "math.hpp"

namespace sm
{
class Renderer
{
  public:
    using Drawer_t = std::function<Color(Vector2, Transform)>;

    Image image;
    Transform transform;

    Renderer(Image image_, size_t thread_count_ = std::thread::hardware_concurrency())
        : image{ image_ }, transform{ image_.dims.as<double>() / 2., 1. }, draw_funcs{},
          thread_count{ thread_count_ }
    {
    }

    template <typename T>
    requires std::invocable<T, Vector2>
    void add_drawer(T&& drawer) { this->draw_funcs.emplace_back(drawer); }

    void render();

  private:
    std::vector<Drawer_t> draw_funcs;
    size_t thread_count;
    std::mutex mut;
};
} // namespace sm
