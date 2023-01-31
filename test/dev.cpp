#include "range/v3/numeric/accumulate.hpp"
#include "range/v3/view/take.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};
    auto ps     = gpu::ParticleSystem{u64(1) << u64(20), Particle<f32>{.pos{0, 0}, .vel{1, 1}}};
    auto rand   = RandomGenerator{};
    for (auto& i : ps.particles.data)
    {
        i.pos    = rand.vector(window.viewport()).cast<f32>();
        i.vel    = rand.polar_vector({0, 4}).cast<f32>();
        i.radius = 0.1F;
    }
    window.view.scale /= 2.0;
    auto watch = Stopwatch{};

    const auto update = [&]
    {
        ps.update();
        print(ps.particles.data.size(), " at ", i32(1000.0 * watch.seconds()), " ms");
        watch.reset();
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 20}, .thickness = 0.03F});
        draw::circle(window, {{2, 3}, 1.4}, {.fill_color{0, 25, 255}});

        for (const auto& particle : ps.particles.data)
        {
            draw::regular_polygon(window, {particle.pos.cast<f64>(), particle.radius}, 3,
                                  {.fill_color = "#ff0000"_c});
        }
        draw::circle(window, {{0, 0}, .1}, {.fill_color{0, 25, 255}});
    };

    run(window, update, draw);
}
