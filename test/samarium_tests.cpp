/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/gui/Window.hpp"
#include "../src/samarium/physics/Dual.hpp"
#include "../src/samarium/physics/collision.hpp"

void App();

int main() { App(); }

void App()
{
    auto rn = sm::Renderer{sm::Image{sm::dims720}};

    const auto gravity      = sm::Vector2{0, -100.0};
    const auto viewport_box = rn.viewport_box();

    auto ball = sm::Dual{sm::Particle{.vel    = {30, 96},
                                      .radius = 2.8,
                                      .mass   = 3.0,
                                      .color  = sm::Color{255, 0, 0}}};
    auto t    = sm::Trail{200};

    sm::util::Stopwatch watch{};
    auto sum = 0.0;

    const auto update = [&](auto delta)
    {
        delta /= 4.0;
        ball->apply_force(ball->mass * gravity);
        ball.update(delta);
        for (auto&& i : viewport_box) sm::phys::collide(ball, i);
    };

    const auto tmp = [](sm::util::Stopwatch& w)
    {
        const auto r = 1.0 / w.time().count();
        fmt::print(stderr, "{:4.2f}\n",
                   r); // print to stderr for no line buffering
        w.reset();
        return r;
    };

    const auto draw = [&]
    {
        rn.fill(sm::Color{16, 18, 20});
        t.push_back(ball.now.pos);
        rn.draw(t, sm::Color{66, 149, 245}, 1.0);
        rn.draw(ball.now);
        rn.render();
        // sum += tmp(watch);
    };

    auto window =
        sm::Window{{.dims = rn.image.dims, .name = "Collision", .framerate = 64}};
    window.run(rn, update, draw, 40, 300);
    // sm::print("Average: ", sum\);
}
