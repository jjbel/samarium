/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/app/App.hpp"
#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"

using namespace sm;

auto main() -> i32
{
    auto iterations = 40UL;

    const auto get_iter = [](Vec2 pos, f64 threshold, u64 max_iterations) -> std::optional<u64>
    {
        auto z                 = std::complex<f64>{};
        const auto pos_complex = to_complex(pos);

        for (auto i : loop::end(max_iterations))
        {
            z = z * z + pos_complex;
            if (std::abs(z) > threshold) { return std::optional<u64>{i}; }
        }

        return std::nullopt;
    };

    const auto colorise =
        [&](Vec2 pos, auto&& gradient, f64 threshold = 42.0, u64 max_iterations = 40)
    {
        if (const auto iter = get_iter(pos, threshold, max_iterations); iter.has_value())
        {
            return gradient(interp::map_range_clamp(
                static_cast<f64>(*iter), {0.0, static_cast<f64>(max_iterations)}, {0.0, 1.0}));
        }
        else { return Color{9, 5, 26}; }
    };

    const auto draw = [&](Vec2 pos) { return colorise(pos, gradients::magma, 42.0, iterations); };

    auto app = App{{.dims = dims720}};
    app.transform.pos += Vec2{.x = 400};
    app.transform.scale *= 40;

    const auto update = [&]
    {
        if (app.mouse.left) { app.transform.pos += app.mouse.current_pos - app.mouse.old_pos; }

        const auto scale = 1.0 + 0.1 * app.mouse.scroll_amount;
        app.transform.scale *= Vec2::combine(scale);
        const auto pos    = app.mouse.current_pos;
        app.transform.pos = pos + scale * (app.transform.pos - pos);
        iterations        = static_cast<u64>(3 * std::log(app.transform.scale.x) + 9);

        app.draw_world_space(draw);
    };

    app.run(update);
}
