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
    auto window = Window{{.dims = {1280, 720}}};
    auto bench  = Benchmark{};

    const auto count = 1'000'000;
    auto ps          = ParticleSystemInstanced(window, count, 0.001F, Color{100, 60, 255});

    auto rand      = RandomGenerator{};
    const auto box = window.world_box(); // TODO gives a square
    for (auto& pos : ps.pos) { pos = rand.vector(box).cast<f32>(); }
    for (auto& vel : ps.vel) { vel = rand.polar_vector({0, 0.1}).cast<f32>(); }

    const auto draw = [&]
    {
        draw::background(Color{});

        const auto mouse_pos = window.pixel2world()(window.mouse.pos).cast<f32>();
        for (auto i : loop::end(ps.size()))
        {
            const auto v = mouse_pos - ps.pos[i];
            const auto l = v.length();
            ps.acc[i]    = v * 0.005F / (l * l * l);
        }

        ps.update();
        ps.draw();
        bench.add("instance draw");

        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{255, 255, 0});

        window.pan();
        window.zoom_to_cursor();
        // std::this_thread::sleep_for(std::chrono::milliseconds(16));
        bench.add_frame();
    };
    run(window, draw);

    bench.print();
    ps.instancer.bench.print();
}
