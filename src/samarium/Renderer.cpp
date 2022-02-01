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

#include "boost/asio/post.hpp"
#include <utility>

#include "Renderer.hpp"

namespace sm
{
void Renderer::render()
{
    const auto funcs = std::move(this->draw_funcs); // make a local copy
    auto s           = std::string("Hello");

    for (size_t i = 0; const auto& chunk : this->chunks)
    {
        const auto task = [&, i, funcs]
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));


            const auto lock = std::lock_guard(mut);
            s += fmt::format("\nthread {:>2}: ", i);
        };

        boost::asio::post(pool, task);

        ++i;
    }

    pool.join();
    draw_funcs = std::vector<Drawer_t>{};
    sm::util::print(s);
}
} // namespace sm
