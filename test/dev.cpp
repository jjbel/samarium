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

constexpr auto ground_height = 100.0;

int main()
{
    auto app            = App{{.dims{1600, 900}}};
    app.transform.scale = {1.0, -1.0}; // + is towards top-right
    app.transform.pos   = {0.0, f64(app.dims().y)}; // origin at bottom left

    auto distance = 0.0;
    auto viewport = app.transformed_bounding_box();
    print(viewport);

    const auto update = [&](f64 dt) {};
    const auto draw   = [&]
    {
        app.fill("#0b0f14"_c);
        app.draw(BoundingBox<f64>{{viewport.min.x, 0.0}, {viewport.max.x, ground_height}},
                 {.fill_color = "#555c66"_c});

        viewport = app.transformed_bounding_box();
    };

    app.run(update, draw);
}
