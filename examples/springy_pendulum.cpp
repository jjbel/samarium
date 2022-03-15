/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"

auto tmp(sm::util::Stopwatch& watch)
{
    // fmt::print(stderr, "\r{:>{}}", "",
    //            sm::util::get_terminal_dims()
    //                .x); // clear line by padding spaces to width of
    //                terminal
    fmt::print(stderr, "{:4.2f}\n",
               1.0 / watch.time().count()); // print to stderr for no line buffering
    watch.reset();
}

int main()
{
    /*
     * SPDX-License-Identifier: MIT
     * Copyright (c) 2022 Jai Bellare
     * See <https://opensource.org/licenses/MIT/> or LICENSE.md
     * Project homepage: https://github.com/strangeQuark1041/samarium
     */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/gui/Window.hpp"
#include "../src/samarium/math/Dual.hpp"
#include "../src/samarium/physics/collision.hpp"

    void App();

    int main() { App(); }

    void App()
    {

        const auto tmp = [](sm::util::Stopwatch& w)
        {
            const auto r = 1.0 / w.time().count();
            fmt::print(stderr, "{:4.2f}\n",
                       r); // print to stderr for no line buffering
            w.reset();
            return r;
        };

        using namespace sm::literals;
        auto rn = sm::Renderer{sm::Image{sm::dims720}};

        const auto gravity = -100.0_y;

        const sm::Vector2 anchor   = 30.0_y;
        const auto rest_length     = 14.0;
        const auto spring_constant = 100.0;

        auto p1 = sm::Particle{
            .pos = {}, .vel = {50, 0}, .radius = 3, .mass = 40, .color = sm::colors::red};
        auto p2 = p1;

        const auto l = sm::LineSegment{{-30, -30}, {30, -9}};

        const auto dims         = rn.image.dims.as<double>();
        const auto viewport_box = rn.viewport_box();
        sm::util::Stopwatch watch{};

        const auto run_every_frame = [&]
        {
            rn.fill(sm::Color{16, 18, 20});
            p1.apply_force(p1.mass * gravity);
            const auto spring = p1.pos - anchor;
            const auto force =
                spring.with_length(spring_constant * (rest_length - spring.length()));
            p1.apply_force(force);
            p1.update();
            auto dual = sm::Dual{p1, p2};
            sm::phys::collide(dual, l);
            for (auto&& i : viewport_box) sm::phys::collide(dual, i);
            std::tie(p1, p2) = std::tuple{dual.now, dual.prev};

            rn.draw_line_segment(l, sm::gradients::blue_green, 0.4);
            rn.draw_line_segment(sm::LineSegment{anchor, p1.pos}, "#c471ed"_c, .6);
            rn.draw(p1);
            p2 = p1;
            rn.render();
            tmp(watch);
        };

        run_every_frame();
        auto window = sm::Window{{rn.image.dims, "Collision", 60}};
        window.run(rn, run_every_frame);
    }
}
