#include "samarium/cuda/cuda.hpp"
#include "samarium/samarium.hpp"


using namespace sm;

auto main(int argc, char* argv[]) -> i32
{
    const auto cell_size = 1.4F;
    // std::strtof(argv[1], nullptr);

    auto window = Window{{.dims = {1000, 1000}}};
    auto bench  = Benchmark{};

    // TODO at 2000, start shooting off
    // keep it fixed time. if fps too low, not enough substeps
    const auto count         = 9'000ULL;
    const auto radius        = 0.3F;
    const auto emission_rate = 6;
    auto sun                 = Vec2f{0, 0};
    const auto grid_color    = Color{255, 255, 255, 90};
    // const auto particle_color = Color{100, 60, 255};
    const auto particle_color = Color{252, 157, 30};

    auto ps = ParticleSystemInstanced<>(window, count, cell_size, radius, particle_color);

    auto rand = RandomGenerator{};
    window.display();
    const auto box = window.world_box(); // TODO gives a square

    for (auto& pos : ps.pos)
    {
        // pos = rand.vector(box).template cast<f32>() * 4.0F;
        pos = rand.polar_vector({0.0, box.max.y}).cast<f32>() +
              rand.polar_vector({0.0, 1}).cast<f32>();
    }

    window.camera.scale /= 30.0;

    // for (auto& vel : ps.vel) { vel = rand.polar_vector({0, 0.1}).template cast<f32>(); }

    const auto gravity = [](Vec2f a, Vec2f b, f32 G, f32 clamp)
    {
        const auto v = a - b;
        const auto l = v.length();
        // gravity:
        auto g = G / (l * l);
        g      = std::min(g, clamp); // clamp the repulsion

        // lennard-jones:
        // const auto r0 = 3 * radius;
        // auto g        = 0.1F * (6 * math::power<6>(r0) / math::power<7>(l) -
        //                   12 * math::power<12>(r0) / math::power<13>(l));
        // g             = std::max(g, -0.1F); // clamp the repulsion
        // nice: if u clamp a lot: only attraction: clumping

        // const auto c = static_cast<u8>(std::abs(g) * 100);
        // const auto c = static_cast<u8>(200);
        // draw::line_segment(window, {ps.pos[i].template cast<f64>(), ps.pos[j].template
        // cast<f64>()},
        //                    Color{c, c, c, 30}, 0.01);

        return (v / l) * g;
    };

    auto image = Image{window.dims};

    auto frame      = 0;
    const auto draw = [&]
    {
        draw::background(Color{255, 255, 255});
        // draw::background(Color{});

        bench.add("bg, display");

        // const auto box = window.world_box();
        // for (auto i : loop::start_end(-100, 100))
        // {
        //     draw::line_segment(window, {{i * cell_size, box.min.y}, {i * cell_size, box.max.y}},
        //                        grid_color, 0.05F);
        //     draw::line_segment(window, {{box.min.x, i * cell_size}, {box.max.x, i * cell_size}},
        //                        grid_color, 0.05F);
        // }
        // bench.add("grid draw");

        const auto mouse_pos = window.pixel2world()(window.mouse.pos).template cast<f32>();
        // for (auto i : loop::end(ps.size()))
        // {
        //     const auto v = mouse_pos - ps.pos[i];
        //     const auto l = v.length();
        //     ps.acc[i]    = v * 0.005F / (l * l * l);
        // }

        // for (auto i : loop::end(emission_rate))
        // {
        //     // ps.pos.push_back({rand.range<f32>({-4, -3.5}), 4.0F});
        //     ps.pos.push_back(
        //         {rand.range<f32>({mouse_pos.x - 0.3F, mouse_pos.x + 0.3F}), mouse_pos.y});

        //     ps.vel.push_back({rand.range<f32>({-0.01F, 0.01F}), rand.range<f32>({-3.6F,
        //     -4.0F})}); ps.acc.push_back({});
        // }
        bench.add("emitter");


        ps.rehash();
        bench.add("rehash");

        for (auto i : loop::end(ps.size())) { ps.acc[i] -= gravity(ps.pos[i], sun, 36.0F, 30.0F); }
        bench.add("sun");

        // auto c = 0;
        // for (auto i : loop::end(ps.size()))
        // {
        //     for (auto j : ps.hash_grid.neighbors(ps.pos[i])) // +1 ?
        //     {
        //         // TODO cud also find for each particle independently, then add up
        //         // twice the looping, but paralellizable
        //         // remember: check i == j, or add a little to l in gravity
        //         if (i >= j) { continue; }
        //         // if (i == j) { continue; }
        //         const auto f = gravity(ps.pos[i], ps.pos[j], 0.0006F, 1.0F);
        //         ps.acc[i] -= f;
        //         ps.acc[j] += f;
        //         // c++;
        //     }
        // }

        // 38.6 fps vs
        cuda::forces({count, ps.pos, ps.acc, nullptr, nullptr, 0.022F, 1.0F});

        bench.add("forces");

        if (frame % 20 == 0)
        {
            // print(c, "/", ps.size() * ps.size() / 2, "   ", ps.size());
            ps.hash_grid.print_occupancy();
        }

        // if (frame > 10000)
        // {
        //     ps.pos.erase(ps.pos.begin(), ps.pos.begin() + emission_rate);
        //     ps.vel.erase(ps.vel.begin(), ps.vel.begin() + emission_rate);
        //     ps.acc.erase(ps.acc.begin(), ps.acc.begin() + emission_rate);
        // }

        bench.add("trimming");

        // ps.self_collision();
        // bench.add("coll");

        ps.update(1.0F / 100.0F);
        bench.add("update");

        ps.draw();
        bench.add("instance draw");

        // draw::circle(window, {sun.template cast<f64>(), 0.9}, Color{255, 255, 0}, 64);
        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{0, 0, 0, 0});

        window.pan();
        window.zoom_to_cursor();
        bench.add_frame();
        frame++;

        window.get_image(image);
        file::write(file::pam, image, fmt::format("./exports/{:05}.pam", frame));

        // if (frame > 54) { window.close(); }
    };
    run(window, draw);

    bench.print();
}
