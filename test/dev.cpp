/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/view/enumerate.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app         = App{{.dims{1600, 800}}};
    const auto count = 200;

    auto plot = std::vector<Vector2>(count);

    auto watch = Stopwatch{};

    auto rand  = RandomGenerator{};
    auto noise = util::PerlinNoise{};

    const auto draw = [&]
    {
        //    for(auto [i, value] : ranges::view)

        app.fill("#16161c"_c);
        app.draw_world_space(
            [&](Vector2 pos)
            {
                const auto thing =
                    noise.detail(pos,
                                 {.scale = 1.0, .detail = 3, .seed = app.frame_counter / 100.0}) *
                    255;
                return Color::from_grayscale(static_cast<u8>(thing));
            });
    };
    // app.run(draw);
    for (auto i : range(20)) { print(rand()); }
}
