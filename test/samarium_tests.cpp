/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <ranges>

#include "gtest/gtest.h"

#include "samarium/samarium.hpp"

#include "tests/Vector2.hpp"

TEST(Main, main)
{
    auto rn = sm::Renderer{sm::Image{{1840, 900}}};

    const auto gravity = sm::Vector2{-100.0};
    auto viewport_box  = rn.viewport_box();

    auto window = sm::Window{rn.image.dims, "Collision", 64};

    auto ball = sm::Dual<sm::Particle>{
        sm::Particle{.vel = {10, 6}, .radius = 4, .color = sm::colors::pink}};
    auto t = sm::Trail{100};

    sm::util::Stopwatch watch{};

    const auto update = [&](auto delta) { ball.update(delta); };

    const auto draw = [&]
    {
        t.push_back(ball.now.pos);
        for (auto&& i : t.span())
        {
            rn.draw(sm::Particle{
                .pos = i, .radius = 1, .color = sm::colors::red.with_alpha(100)});
        }
        rn.draw(ball.now);

        // fmt::print(stderr, "\r{:>{}}", "",
        //            sm::util::get_terminal_dims()
        //                .x); // clear line by padding spaces to width of terminal
        fmt::print(stderr, "Current framerate: {}\n",
                   watch.time().count() * 1000); // print to stderr for no line buffering
        watch.reset();
    };

    window.run(rn, sm::Color(12, 12, 20), update, draw);
}
