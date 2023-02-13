#include <array>
#include <vector>

#include "samarium/gl/draw/grid.hpp"
#include "samarium/gl/draw/shapes.hpp"
#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

static constexpr auto particle_colors = std::to_array<Color>(
    {"#fb2238"_c, "#0771F2"_c, "#11F762"_c, "#FCDD45"_c, "#7F11CE"_c, "#F4661A"_c});

struct MyParticle
{
    Vector2 pos{};
    Vector2 vel{};
    i32 color{};
};

struct Interaction
{
    f64 max_distance{4.0};
    f64 equilibrium_length{1.0 / 2.0};
    f64 repulsion{2.0};

    Grid<f64> peaks{{3, 3}, 1.0};

    auto force(i32 on, i32 due_to, f64 distance)
    {
        distance /= max_distance;
        if (distance <= equilibrium_length)
        {
            return interp::lerp<f64>(distance / equilibrium_length, {-repulsion, 0.0});
        }
        if (distance >= 1) { return 0.0; }
        return interp::map_range<f64, f64>(math::abs(distance - (equilibrium_length + 1.0) / 2.0),
                                           {0, (1.0 - equilibrium_length) / 2.0},
                                           {peaks[Indices::make(due_to, on)], 0.0});
    }

    auto operator()(MyParticle& current, const MyParticle& other)
    {
        const auto radius_vector = other.pos - current.pos;
        const auto bond_force    = force(current.color, other.color, radius_vector.length());
        current.vel += 0.01 * radius_vector.with_length(bond_force);
        return bond_force;
    }
};

auto main() -> i32
{
    auto window           = Window{{{1920, 1080}}};
    auto rand             = RandomGenerator{};
    const auto substeps   = 8;
    const auto delta_time = 0.01 / substeps;
    const auto friction   = 1.0;
    auto interaction      = Interaction{.peaks{std::vector<f64>{2, 0, //
                                                                0, 0},
                                          Dimensions{2, 2}}};
    const auto points     = rand.poisson_disc_points(0.8, window.viewport(), 16);
    auto particles        = std::vector<MyParticle>(points.size());
    auto frame_count      = u64(0);
    print(points.size());

    for (auto i : loop::end(particles.size()))
    {
        auto& particle = particles.at(i);
        particle.pos   = points.at(i) * 0.8;
        particle.vel   = rand.polar_vector({0, 0});
        // particle.color = particle.pos.x < 0 ? 1 : 0;
        particle.color = rand.range<i32>({0, 2});
    }

    const auto update = [&]
    {
        draw::background("#12121a"_c);
        for (auto _ : loop::end(substeps))
        {
            for (auto& i : particles)
            {
                i.vel *= 1 - friction / 100.0;
                i.pos += i.vel * delta_time;
            }

            auto grid = HashGrid<u64>{interaction.max_distance * 1.5};
            for (auto i : loop::end(particles.size())) { grid.insert(particles.at(i).pos, i); }
            for (auto i : loop::end(particles.size()))
            {
                auto& current = particles.at(i);
                for (auto j : grid.neighbors(current.pos))
                {
                    if (i == j) { continue; }
                    const auto& other = particles.at(j);
                    const auto force  = interaction(current, other);
                    // if (force != 0.0)
                    // {
                    //     auto bond_color = colors::crimson;
                    //     if (force > 0) { bond_color = colors::springgreen; }
                    //     draw::line_segment(window, {current.pos, other.pos},
                    //                        bond_color.with_multiplied_alpha(0.1), 0.06);
                    // }
                }
                // const auto radius_vector = window.mouse.pos - current.pos;
                // current.vel += radius_vector.with_length(
                //     interaction.force(current.color, 0, radius_vector.length()));
            }
        }

        for (const auto& i : particles)
        {
            draw::circle(window, {i.pos, 0.11},
                         {.fill_color = particle_colors.at(static_cast<u64>(i.color))});
            // draw::circle(window, {i.pos, interaction.max_distance},
            //              {.fill_color = "#a5d5ff"_c.with_multiplied_alpha(0.01)});
        }
    };

    run(window, update);
}
