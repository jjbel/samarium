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

    const auto circle      = Circle{{0, 0}, 0.0001};
    const auto color       = Color{100, 60, 255};
    const auto point_count = 3;
    const auto points      = math::regular_polygon_points<f32>(point_count, circle);

    const auto count = 10'000'000;
    /*
    conclusion:
    for 64 verts:
    window.display() and draw instance take:
    100K particles: 5.1ms, 0.3ms
      1M particles: 28-55ms, 2ms
     10M particles: 440ms, 45ms

    for 3 verts:
    10M particles: 19ms, 19ms

    avoid copying to instancer.instances_pos
    (hence use data oriented design: store pos in instancer.instances_pos only)
    copying from instancer.instances_pos to buffer is unavoidable
    no need to copy to/from instancer.geometry

    TODO: use gl classes
    TODO: have to set vertattrib every call?
    */

    auto rand      = RandomGenerator{};
    auto pts       = std::vector<Vector2f>();
    const auto box = window.world_box(); // TODO gives a square
    for (auto i : loop::end(count)) { pts.push_back(rand.vector(box).cast<f32>()); }
    auto instancer = gl::Instancer(window, points, pts);

    while (true)
    {
        bench.reset();
        if (!window.is_open()) { break; }
        bench.add("is_open");


        draw::background(Color{});
        bench.add("bg");
        const auto disp = rand.polar_vector({0, 0.001}).cast<f32>();
        bench.add("rand disp");
        // rand is the bottleneck, so just add const vec
        // for (auto& pt : pts) { pt += rand.polar_vector({0, 0.001}).cast<f32>(); }

        // for (auto& pt : pts) { pt += disp; }
        // instancer.instances_pos = pts;

        instancer.draw(color);
        bench.add("instance draw");

        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{255, 255, 0});
        bench.add("circle draw");

        window.pan();
        window.zoom_to_cursor();
        bench.add("pan zoom");
        // std::this_thread::sleep_for(std::chrono::milliseconds(16));
        bench.add("sleep");
        window.display();
        bench.add("display");
        bench.add_frame();
    }

    bench.print();
    instancer.bench.print();
}
