// #include "samarium/cuda/cuda.hpp"
// #include "samarium/cuda/HashGrid.hpp"
// #include "samarium/samarium.hpp"


// using namespace sm;

// int main()
// {
//     print("Hello!");
//     const auto width  = u64(5);
//     const auto height = u64(5);

//     auto hg = cuda::HashGrid{width, height};

//     for (auto x : loop::start_end(-width, width))
//     {
//         for (auto y : loop::start_end(-height, height)) { print(hg.get_index({x, y})); }
//     }
// }

#include "samarium/graphics/colors.hpp"
#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"

using namespace sm;

auto main(int argc, char* argv[]) -> i32
{
    const auto cell_size = 0.3F;
    auto window          = Window{{.dims = {1920, 1080}}};
    auto bench           = Benchmark{}; // to see how fast it runs

    const auto count         = 0;
    const auto radius        = 0.01F;
    const auto emission_rate = 6;
    auto sun                 = Vec2f{0, 0};

    auto ps = ParticleSystemInstanced<>(window, count, cell_size, radius, Color{100, 60, 255});

    auto rand = RandomGenerator{};
    window.display();
    window.camera.scale /= 7.0; // zoom out


    const auto gravity = [](Vec2f a, Vec2f b, f32 G, f32 clamp)
    {
        const auto v = a - b;
        const auto l = v.length();
        // gravity:
        auto g = G / (l * l);
        g      = std::min(g, clamp); // prevent the repulsive force from being too strong
        return (v / l) * g;
    };

    auto max_occ = 0ULL;

    auto frame      = 0;
    const auto draw = [&]
    {
        draw::background(Color{});

        const auto box  = window.world_box();
        const auto box1 = window.world_box(); // TODO gives a square

        auto current_max_occ = u64{};
        auto& map            = ps.hash_grid.map;
        for (const auto& [key, value] : map)
        {
            current_max_occ = std::max(current_max_occ, value.size());
        }

        for (auto i : loop::start_end(-100, 100))
        {
            for (auto j : loop::start_end(-100, 100))
            {
                const auto it = map.find(Vec2_t<i32>::make(i, j));
                if (it == map.end()) { continue; }
                // TODO why into 4
                const auto w     = cell_size;
                const auto verts = std::array<Vec2f, 4>{
                    Vec2f{i * w, j * w}, Vec2f{i * w, (j + 1) * w}, Vec2f{(i + 1) * w, (j + 1) * w},
                    Vec2f{(i + 1) * w, j * w}};
                const auto gradient = gradients::magma;
                draw::polygon(
                    window, verts,
                    gradient(std::min(static_cast<f64>(it->second.size()) / current_max_occ, 1.0)));
                max_occ = std::max(max_occ, it->second.size());
            }
        }

        auto mouse_pos = window.pixel2world()(window.mouse.pos).cast<f32>();

        for (auto i : loop::end(emission_rate))
        {
            ps.pos.push_back(
                {rand.range<f32>({mouse_pos.x - 0.1F, mouse_pos.x + 0.1F}), mouse_pos.y});

            ps.vel.push_back(
                {rand.range<f32>({-0.00001F, 0.00001F}), rand.range<f32>({3.999F, 4.0F})});
            ps.acc.push_back({});
        }

        ps.rehash();

        for (auto i : loop::end(ps.size())) { ps.acc[i] -= gravity(ps.pos[i], sun, 36.0F, 30.0F); }

        for (auto i : loop::end(ps.size()))
        {
            for (auto j : ps.hash_grid.neighbors(ps.pos[i])) // +1 ?
            {
                if (i >= j) { continue; }
                const auto f = gravity(ps.pos[i], ps.pos[j], 0.00008F, 1.0F);
                ps.acc[i] -= f;
                ps.acc[j] += f;
            }
        }

        if (frame > 500)
        {
            ps.pos.erase(ps.pos.begin(), ps.pos.begin() + emission_rate);
            ps.vel.erase(ps.vel.begin(), ps.vel.begin() + emission_rate);
            ps.acc.erase(ps.acc.begin(), ps.acc.begin() + emission_rate);
        }

        ps.update(1.0F / 100.0F);
        ps.draw();

        draw::circle(window, {sun.cast<f64>(), 9.4}, Color{255, 255, 0}, 64);
        draw::circle(window, {{0.1, 0.2}, 0.1}, Color{0, 0, 0, 0});

        window.pan();
        window.zoom_to_cursor();
        frame++;
    };
    run(window, draw);
}

