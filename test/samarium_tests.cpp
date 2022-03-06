/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "catch2/catch_test_macros.hpp"

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"
#include "tests/Vector2.hpp"
#include "tests/concepts.hpp"

auto tmp(sm::util::Stopwatch& watch)
{
    // fmt::print(stderr, "\r{:>{}}", "",
    //            sm::util::get_terminal_dims()
    //                .x); // clear line by padding spaces to width of
    //                terminal
    fmt::print(stderr, "{:4.2f}\n",
               1.0 /
                   watch.time().count()); // print to stderr for no line buffering
    watch.reset();
}

auto App()
{
    auto rn = sm::Renderer{sm::Image{sm::dims720}};

    const auto gravity      = sm::Vector2{0, -100.0};
    const auto viewport_box = rn.viewport_box();

    auto ball = sm::Dual{sm::Particle{
        .vel = {30, 96}, .radius = 2.8, .color = sm::Color{255, 0, 0}}};
    auto t    = sm::Trail{200};

    sm::util::Stopwatch watch{};

    const auto update = [&](auto delta)
    {
        delta /= 4.0;
        ball->apply_force(ball->mass * gravity);
        ball.update(delta);
        for (auto&& i : viewport_box) sm::phys::collide(ball, i);
    };

    const auto draw = [&]
    {
        // rn.fill(sm::Color{16, 18, 20});
        // t.push_back(ball.now.pos);
        // rn.draw(t, sm::colors::orange, 1.0);
        // rn.draw(ball.now);
        tmp(watch);
    };

    auto window =
        sm::Window{{.dims = rn.image.dims, .name = "Collision", .framerate = 64}};
    window.run(rn, update, draw, 40);
    // sm::file::export_to(rn.image, "temp.tga");
}

TEST_CASE("App", "main") { App(); }
