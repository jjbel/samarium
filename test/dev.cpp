/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;

auto main(int argc, char* argv[]) -> i32
{
    // TODO with 0.1, why does it still draw so many lines?
    const auto cell_size = 1.4F;
    // const auto cell_size = std::strtof(argv[1], nullptr);
    // print(cell_size);
    // std::abort();

    auto window = Window{{.dims = {1920, 1080}}};
    auto bench  = Benchmark{};

    // TODO at 2000, start shooting off
    // keep it fixed time. if fps too low, not enough substeps
    const auto count  = 10'000;
    const auto radius = 0.01F;


    auto ps = ParticleSystemInstanced<>(window, count, cell_size, radius, Color{100, 60, 255});

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
        bench.add("bg, display");

        const auto box = window.world_box();
        for (auto i : loop::start_end(-100, 100))
        {
            draw::line_segment(window, {{i * cell_size, box.min.y}, {i * cell_size, box.max.y}},
                               Color{255, 255, 0, 90}, 0.01F);
            draw::line_segment(window, {{box.min.x, i * cell_size}, {box.max.x, i * cell_size}},
                               Color{255, 255, 0, 90}, 0.01F);
        }
        bench.add("grid draw");

        // const auto mouse_pos = window.pixel2world()(window.mouse.pos).cast<f32>();
        // for (auto i : loop::end(ps.size()))
        // {
        //     const auto v = mouse_pos - ps.pos[i];
        //     const auto l = v.length();
        //     ps.acc[i]    = v * 0.005F / (l * l * l);
        // }

        ps.rehash();
        bench.add("rehash");

        auto c = 0;
        for (auto i : loop::end(ps.size()))
        {
            for (auto j : ps.hash_grid.neighbors(ps.pos[i])) // +1 ?
            {
                // TODO cud also find for each particle independently, then add up
                // twice the looping, but paralellizable
                // commenting this and (ps.acc[j]+=f) breaks it
                if (i >= j) { continue; }

                const auto v = ps.pos[i] - ps.pos[j];
                const auto l = v.length();
                // gravity:
                auto g = 0.0006F / (l * l);
                g      = std::min(g, 1.0F); // clamp the repulsion

                // lennard-jones:
                // const auto r0 = 3 * radius;
                // auto g        = 0.1F * (6 * math::power<6>(r0) / math::power<7>(l) -
                //                   12 * math::power<12>(r0) / math::power<13>(l));
                // g             = std::max(g, -0.1F); // clamp the repulsion
                // nice: if u clamp a lot: only attraction: clumping

                // const auto c = static_cast<u8>(std::abs(g) * 100);
                // const auto c = static_cast<u8>(200);
                // draw::line_segment(window, {ps.pos[i].cast<f64>(), ps.pos[j].cast<f64>()},
                //                    Color{c, c, c, 30}, 0.01);

                const auto f = (v / l) * g;
                ps.acc[i] -= f;
                // ps.acc[j] += f;
                c++;
            }
        }
        if (frame % 3 == 0)
        {
            print(c, "/", count * count / 2);
            ps.hash_grid.print_occupancy();
        }
        bench.add("forces");

        // ps.self_collision();
        // bench.add("coll");

        ps.update(1.0F / 100.0F);
        bench.add("update");

        ps.draw();
        bench.add("instance draw");

        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{0, 0, 0, 0});

        window.pan();
        window.zoom_to_cursor();
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        bench.add_frame();
        frame++;
        // file::write(file::pam, window.get_image(), fmt::format("./exports/{:05}.pam", frame));
    };
    run(window, draw);

    bench.print();
    // ps.instancer.bench.print();
}
