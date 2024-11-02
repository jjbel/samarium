/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;

auto main() -> i32
{
    auto window = Window{{.dims = {1920, 1080}}};
    auto bench  = Benchmark{};

    // TODO at 2000, start shooting off
    const auto count  = 1000;
    const auto radius = 0.06F;
    auto ps           = ParticleSystemInstanced(window, count, radius, Color{100, 60, 255});

    auto rand = RandomGenerator{};
    window.display();
    const auto box = window.world_box(); // TODO gives a square
    for (auto& pos : ps.pos) { pos = rand.vector(box).cast<f32>() * 4.0F; }
    // for (auto& vel : ps.vel) { vel = rand.polar_vector({0, 0.1}).cast<f32>(); }
    window.camera.scale /= 5.0;
    auto frame      = 0;
    const auto draw = [&]
    {
        draw::background(Color{});

        // const auto mouse_pos = window.pixel2world()(window.mouse.pos).cast<f32>();
        // for (auto i : loop::end(ps.size()))
        // {
        //     const auto v = mouse_pos - ps.pos[i];
        //     const auto l = v.length();
        //     ps.acc[i]    = v * 0.005F / (l * l * l);
        // }

        for (auto i : loop::end(ps.size()))
        {
            for (auto j : loop::end(i)) // +1 ?
            {
                const auto v = ps.pos[i] - ps.pos[j];
                const auto l = v.length();
                // gravity:
                // const auto f = v * 0.000005F / (l * l * l);

                // lennard-jones:
                const auto r0 = 3 * radius;
                auto g        = 0.04F * (6 * std::pow(r0, 6.0F) / std::pow(l, 7.0F) -
                                  12 * std::pow(r0, 12.0F) / std::pow(l, 13.0F));
                g             = std::max(g, -0.1F); // clamp the repulsion
                // nice: if u clamp a lot: only attraction: clumping

                // const auto c = static_cast<u8>(std::abs(g) * 100);
                // draw::line_segment(window, {ps.pos[i].cast<f64>(), ps.pos[j].cast<f64>()},
                //                    Color{c, c, c}, 0.7);

                const auto f = (v / l) * g;
                ps.acc[i] -= f;
                ps.acc[j] += f;
            }
        }

        ps.update();
        ps.draw();
        bench.add("instance draw");

        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{0, 0, 0, 0});

        window.pan();
        window.zoom_to_cursor();
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        bench.add_frame();
        frame++;
        file::write(file::pam, window.get_image(), fmt::format("./exports/{:05}.pam", frame));
    };
    run(window, draw);

    bench.print();
    ps.instancer.bench.print();
}
