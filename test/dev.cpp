#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};

    auto rand = RandomGenerator{};

    auto ps = gpu::ParticleSystem(8, {.pos{69, 42}, .vel{4, 5}});

    const auto print_buf = [&]
    {
        for (auto i : ps.particles.data) { fmt::print("{}, ", i.pos); }
        print();
    };

    for (auto& i : ps.particles.data)
    {
        i.pos    = rand.vector(window.viewport()).cast<f32>();
        i.vel    = rand.polar_vector({0, 4}).cast<f32>();
        i.radius = 0.8F;
    }

    // ================================== NOTE ========================================
    // reading from the gpu seems to take 16ms whether we use regular buffers or persistent mapped
    // buffers however with persistent mapped buffers only the first element seems to be updated
    // perhaps use triple buffering? (seep
    // https://www.cppstories.com/2015/01/persistent-mapped-buffers-benchmark/)

    print_buf();
    ps.update();
    print_buf();
    ps.update();
    print_buf();

    window.view.scale /= 2.0;

    auto watch = Stopwatch{};

    const auto update = [&]
    {
        watch.reset();
        ps.update();
        // for (auto& i : ps.particles()) { i.update(); }
        watch.print();
        // print(ps.buffers.particles.data[0]);
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 20}, .thickness = 0.03F});
        // draw::grid_lines(window, {.spacing = 4, .color{255, 255, 255, 20}, .thickness = 0.08F});
        draw::circle(window, {{2, 3}, 1.4}, {.fill_color{0, 25, 255}});

        for (const auto& particle : ps.particles.data)
        {
            draw::regular_polygon(window, {particle.pos.cast<f64>(), particle.radius}, 8,
                                  {.fill_color = "#ff0000"_c});
        }
        draw::circle(window, {{0, 0}, .1}, {.fill_color{0, 25, 255}});
    };

    run(window, update, draw);
    print(sizeof(Particle<f32>));

    print_buf();
}

// old texture stuff:
// auto texture = gl::Texture{gl::ImageFormat::R32F};
// imageStore( data, pos, vec4( in_val, 0.0, 0.0, 0.0 ) );
// buffer.bind_level(0, 0, gl::Access::ReadWrite);
