#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};

    auto rand = RandomGenerator{};

    auto ps = gpu::ParticleSystem(20, {.pos{69, 42}, .vel{4, 5}});
    print(ps.particles[0]);

    for (auto& i : ps.particles)
    {
        i.pos    = rand.vector(window.viewport()).cast<f32>();
        i.vel    = rand.polar_vector({0, 4}).cast<f32>();
        i.radius = 0.8F;
    }

    print(ps.particles[0]);
    ps.update();
    print(ps.particles[0]);

    window.view.scale /= 2.0;

    auto watch = Stopwatch{};

    const auto update = [&]
    {
        watch.reset();
        ps.update();
        // for (auto& i : ps.particles()) { i.update(); }
        // watch.print();
        // print(ps.buffers.particles.data[0]);
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 20}, .thickness = 0.03F});
        draw::grid_lines(window, {.spacing = 4, .color{255, 255, 255, 20}, .thickness = 0.08F});
        for (const auto& particle : ps.particles)
        {
            draw::regular_polygon(window, {particle.pos.cast<f64>(), particle.radius}, 8,
                                  {.fill_color = "#ff0000"_c});
        }
        draw::circle(window, {{0, 0}, 1}, {.fill_color{0, 25, 255}});
    };

    run(window, update, draw);

    // print(ps.particles()[0]);
}

// old texture stuff:
// auto texture = gl::Texture{gl::ImageFormat::R32F};
// imageStore( data, pos, vec4( in_val, 0.0, 0.0, 0.0 ) );
// buffer.bind_level(0, 0, gl::Access::ReadWrite);
