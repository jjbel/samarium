/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/action/remove_if.hpp"
#include "range/v3/algorithm/contains.hpp"
#include "range/v3/algorithm/remove_copy_if.hpp"

using namespace sm;
using namespace sm::literals;

struct Platform
{
    i32 from{};
    i32 to{};
    i32 height{};

    auto line_segment() const
    {
        return LineSegment{{f64(from), f64(height)}, {f64(to), f64(height)}};
    }
};

int main()
{
    struct Params
    {
        f64 speed  = 4.0;
        f64 radius = 0.5;
    } params;

    auto app = App{{.dims = {1800, 900}}};
    app.transform.scale *= 6;
    app.transform.pos = app.dims().as<f64>() * Vector2{0.1, 0.85};

    auto player = Particle{.pos = {0.0, 2.0}, .radius = params.radius};
    auto trail  = Trail{50};
    auto jump   = Keyboard::OnKeyDown{{Keyboard::Key::Space},
                                    [&]
                                    {
                                        player.vel.y += 12;
                                        print("Jump");
                                    }};

    auto rand = RandomGenerator{128, RandomMode::NonDeterministic};

    auto platforms = std::vector<Platform>();

    const auto add_platform = [&]
    {
        const auto box = app.transformed_bounding_box();
        auto platform  = Platform{};
        platform.from  = rand.range<i32>(
            {static_cast<i32>(box.min.x + box.max.x), static_cast<i32>(2 * box.max.x)});
        platform.to = rand.range<i32>(
            {static_cast<i32>(box.min.x + box.max.x), static_cast<i32>(2 * box.max.x)});
        if (platform.from > platform.to) { std::swap(platform.from, platform.to); }
        platform.height =
            rand.range<i32>({static_cast<i32>(box.min.y), static_cast<i32>(box.max.y)});
        // if (platforms.empty())
        // {
        //     print("Hallo");
        //     platform.height =
        //         rand.range<i32>({static_cast<i32>(box.min.y), static_cast<i32>(box.max.y)});
        //     platforms.push_back(platform);
        //     return;
        // }

        // print("Haaallo");
        // auto height = -1;
        // while (ranges::contains(platforms, height, &Platform::height))
        // {
        //     platform.height =
        //         rand.range<i32>({static_cast<i32>(box.min.y), static_cast<i32>(box.max.y)});
        // }

        // platform.height = height;
        platforms.push_back(platform);
    };

    for (auto i : range(8)) { add_platform(); }

    const auto update = [&](f64 dt)
    {
        const auto viewport = app.transformed_bounding_box();
        jump();
        // app.zoom_pan();
        app.transform.pos.x -= params.speed * dt * app.transform.scale.x;

        player.acc = Vector2{0.0, -30};

        player.update(dt);
        player.pos.x += params.speed * dt;

        if (player.pos.y <= player.radius)
        {
            player.pos.y = player.radius;
            player.vel.y = 0;
        }

        if (player.pos.y + player.radius > viewport.max.y) { std::exit(0); }

        trail.push_back(player.pos);

        platforms |= ranges::actions::remove_if([viewport](const Platform& platform)
                                                { return platform.from < viewport.min.x; });

        print(platforms.size());
    };

    const auto draw = [&]
    {
        app.fill("#151726"_c);
        // app.draw(App::GridLines{.scale = 1.0, .line_thickness = 0.01, .axis_thickness = 0.016});
        app.draw(App::GridDots{.scale = 1.0, .thickness = 0.03});
        app.draw(trail, "#17ff70"_c, 0.1, 1.0);
        app.draw(Circle{player.pos, player.radius},
                 {.fill_color = "#ff6c17"_c, .border_width = 0.1});

        for (const auto& platform : platforms)
        {
            const auto box = BoundingBox<f64>::from_centre_width_height(
                Vector2{f64(platform.from + platform.to) / 2.0, f64(platform.height)},
                f64(platform.to - platform.from), 0.2);
            app.draw(box, {.fill_color = "#808080"_c, .border_width = 0.1});
        }
    };

    print();
    app.run(update, draw);
    // auto vec = StaticVector<int, 64>({3, 6, 8, 9});
    // for (auto i : vec) { print(i); }
}
