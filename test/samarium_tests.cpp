#include <ranges>

#include "samarium/samarium.hpp"

int main()
{
    using namespace sm::literals;
    auto rn = sm::Renderer{sm::Image{{1840, 900}}};

    const auto gravity = -100.0_y;

    auto viewport_box = rn.viewport_box();

    auto window = sm::Window{rn.image.dims, "Collision", 64};

    const auto count = 100;
    auto now         = std::vector(
                count,
                sm::Particle{.radius = 1.6, .mass = 4, .color = sm::Color{255, 12, 53}});
    auto prev = now;

    for (auto& p : now)
    {
        p.pos = 0.9 * sm::random::rand_vector(
            rn.transform.apply_inverse(rn.image.rect().as<double>()));
    }

    for (auto& p : now)
    {
        if (p.pos.x > -40)
        {
            p.vel =
                sm::random::rand_vector(sm::Extents<double>{30, 84},
                                        sm::Extents<double>{0, 360.0_degrees});
            p.mass   = 0.15;
            p.radius = 0.8;
            p.color  = sm::Color{100, 100, 255};
        }
        else
        {
            p.vel =
                sm::random::rand_vector(sm::Extents<double>{0, 10},
                                        sm::Extents<double>{0, 360.0_degrees});
            p.mass   = 5;
            p.radius = 4;
        }
    }

    for (int i = 0; i < 10; i++) sm::print(sm::gradients::heat(i / 10.0));

    sm::util::Stopwatch watch{};

    const auto run_every_frame = [&]
    {
        for (size_t i = 0; i < count; i++)
        {
            auto& p_now  = now[i];
            auto& p_prev = prev[i];
            p_now.apply_force(p_now.mass * gravity);
            // viewport_box[0].translate(0.001_x);

            sm::update(p_now);

            for (auto& p : now)
                if (&p != &p_now) sm::phys::collide(p_now, p);

            for (auto&& line : viewport_box)
                sm::phys::collide(p_now, p_prev, line);

            rn.draw(p_now);
            // rn.draw_line_segment(viewport_box[0]);
        }
        prev = now;

        fmt::print(stderr, "\r{:>{}}", "",
                   sm::util::get_terminal_dims()
                       .x); // clear line by padding spaces to width of terminal
        fmt::print(
            stderr, "\rCurrent framerate: {}",
            std::round(
                1.0 /
                watch.time().count())); // print to stderr for no line buffering
        watch.reset();
    };

    window.run(rn, sm::Color(12, 12, 20), run_every_frame);
}
