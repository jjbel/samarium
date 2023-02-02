#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};
    auto ps     = gpu::ParticleSystem{u64(1) << u64(22), Particle<f32>{.pos{0, 0}, .vel{1, 1}}, 16};
    auto rand   = RandomGenerator{};
    for (auto& i : ps.particles.data)
    {
        i.pos    = rand.vector(window.viewport()).cast<f32>() / 2.0F;
        i.vel    = rand.polar_vector({0, 4}).cast<f32>();
        i.radius = 0.1F;
    }

    auto watch = Stopwatch{};

    const auto update = [&]
    {
        ps.update();

        draw::background("#111114"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 20}, .thickness = 0.03F});

        ps.draw(window, "#2050d6"_c.with_multiplied_alpha(0.4), 0.4F, 3);
    };

    while (window.is_open())
    {
        watch.reset();
        window.display(); // glfwSwapBuffers takes 66ms
        update();
        watch.print();
    }
}
