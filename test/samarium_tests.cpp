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

    const auto viewport_box = rn.viewport_box();

    auto window = sm::Window{rn.image.dims, "Collision", 60};

    const auto count = 100;
    auto now         = std::vector(count, sm::Particle{.radius = 1.5, .mass = 4});
    auto prev        = now;

    for (auto& p : now)
    {
        p.pos = sm::random::rand_vector(
            rn.transform.apply_inverse(rn.image.rect().as<double>()));
        p.vel = sm::random::rand_vector(sm::Extents<double>{0, 24},
                                        sm::Extents<double>{0, 360.0_degrees});
    }

    // for (int i = 0; i < 10; i++) sm::util::print(sm::gradients::heat(i
    // / 10.0));

    sm::util::Stopwatch watch{};


    const auto run_every_frame = [&]
    {
        for (size_t i = 0; i < count; i++)
        {
            auto& p_now  = now[i];
            auto& p_prev = prev[i];
            // p_now.apply_force(p_now.mass * gravity);

            sm::update(p_now);

            for (auto& p : now)
                if (&p != &p_now) sm::phys::collide(p_now, p);

            for (auto&& line : viewport_box)
                sm::phys::collide(p_now, p_prev, line);

            rn.draw(p_now, sm::gradients::heat(p_now.vel.length() / 24.0));
        }
        prev = now;

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
