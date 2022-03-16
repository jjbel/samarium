/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/gui/Window.hpp"
#include "../src/samarium/gui/sfml.hpp"
#include "../src/samarium/math/Dual.hpp"
#include "../src/samarium/physics/collision.hpp"
#include "../src/samarium/util/file.hpp"

using sm::print;

void App();

int main() { App(); }

void App()
{
    auto rn = sm::Renderer{sm::Image{sm::dims720}};


    const auto gravity      = sm::Vector2{0, -100.0};
    const auto viewport_box = rn.viewport_box();

    auto ball = sm::Dual{
        sm::Particle{.vel = {30, 96}, .radius = 2.8, .mass = 3.0, .color = sm::Color{255, 0, 0}}};
    auto t = sm::Trail{200};

    const auto update = [&](auto delta)
    {
        // delta /= 4.0;
        ball->apply_force(ball->mass * gravity);
        ball.update(delta);
        for (auto&& i : viewport_box) sm::phys::collide(ball, i);
    };

    auto window = sm::Window{{.dims = rn.image.dims, .name = "Collision", .framerate = 64}};

    const auto draw = [&]
    {
        rn.fill(sm::Color{16, 18, 20});
        t.push_back(ball.now.pos);
        rn.draw(t, sm::gradients::rainbow, 0.3);
        rn.draw(ball.now);
        rn.render();

        if (window.mouse.left)
        {
            const auto mouse_pos = rn.transform.apply_inverse(window.mouse.pos.now);
            if (sm::math::distance_sq(mouse_pos, ball.now.pos) <= ball->radius * ball->radius)
                ball.now.pos += window.mouse.vel() / rn.transform.scale;
        }

        if (window.mouse.middle) { rn.transform.pos += window.mouse.vel().as<sm::f64>(); }

        fmt::print(stderr, "\nFrame {}: ", window.frame_counter);
    };

    window.run(rn, update, draw, 40, 700);
    sm::file::export_tga(rn.image);
}
