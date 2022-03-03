/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "catch2/catch_test_macros.hpp"

#include "samarium/samarium.hpp"
#include "samarium/util/ostream.hpp"
#include "tests/Vector2.hpp"
#include "tests/concepts.hpp"

auto App()
{
    auto rn = sm::Renderer{sm::Image{{1000, 500}}};

    const auto gravity = sm::Vector2{-100.0};
    auto viewport_box  = rn.viewport_box();

    auto window = sm::Window{rn.image.dims, "Collision", 64};

    auto ball = sm::Dual<sm::Particle>{sm::Particle{
        .vel = {10, 6}, .radius = 9.9, .color = sm::Color{255, 100, 150}}};
    auto t    = sm::Trail{100};

    sm::util::Stopwatch watch{};

    const auto update = [&](auto delta) { ball.update(delta); };

    const auto draw = [&]
    {
        t.push_back(ball.now.pos);
        for (auto&& i : t.span())
        {
            rn.draw(sm::Particle{.pos    = i,
                                 .radius = 1,
                                 .color  = sm::Color{255, 0, 0}.with_alpha(110)});
        }
        rn.draw(ball.now);

        // fmt::print(stderr, "\r{:>{}}", "",
        //            sm::util::get_terminal_dims()
        //                .x); // clear line by padding spaces to width of
        //                terminal
        fmt::print(stderr, "Current framerate: {}\n",
                   watch.time().count() *
                       1000); // print to stderr for no line buffering
        watch.reset();
    };
    std::cout << ball.now.pos;
    rn.draw(ball.now);
    // window.run(rn, sm::Color(12, 12, 20), update, draw);
}

TEST_CASE("App", "main") { App(); }
