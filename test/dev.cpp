/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app   = App{{.dims{1800, 900}}};
    auto watch = Stopwatch{};

    const auto draw = [&]
    {
        app.fill("#0D0D13"_c);

        if (app.mouse.left) { app.transform.pos += app.mouse.current_pos - app.mouse.old_pos; }

        const auto scale = 1.0 + 0.1 * app.mouse.scroll_amount;
        app.transform.scale *= Vector2::combine(scale);
        const auto pos    = app.mouse.current_pos;
        app.transform.pos = pos + scale * (app.transform.pos - pos);

        app.draw(App::GridLines{.levels = 1});
        app.draw(App::GridDots{});
    };

    app.run(draw);
}
