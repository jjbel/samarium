/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <concepts>

#include "samarium/gui/Window.hpp"
#include "samarium/util/Stopwatch.hpp"

namespace sm
{
/**
 * @brief               call update  -> display the window
 *
 * @param  window       Window to display
 * @param  update       Callable which updates the state of objects
 */
auto run(Window& window, const std::invocable auto& update)
{
    while (window.is_open())
    {
        update();
        window.display();
    }
}

/**
 * @brief               call update  -> call draw -> display the window
 *
 * @param  window       Window to display
 * @param  update       Callable which updates the state of objects
 * @param  draw         Callable which draws objects
 * @param  substeps     Call update this many times for better accuracy. Divide \f$ \Delta t \f$
 * accordingly
 */
auto run(Window& window,
         const auto& update, // std::invocable<void(f64)> ?
         const auto& draw,
         u64 substeps = 1)
{
    auto watch = Stopwatch{};
    auto dt    = 0.0;
    while (window.is_open())
    {
        dt = watch.seconds();
        // clamp dt so if there s a big lag it doesnt explode
        watch.reset();

        for (auto i : loop::end(substeps))
        {
            // TODO is this correct dt?
            // https://youtu.be/yGhfUcPjXuE?t=333 use diff bw n-1 and n-2
            update(dt / substeps);
            // std::ignore = i;
        }
        draw();
        window.display();
    }
}

/**
 * @brief               call update  -> call draw -> display the window
 *
 * @param  window       Window to display
 * @param  update       Callable which updates the state of objects
 * @param  draw         Callable which draws objects
 * @param  substeps
 * @param  dt           Fixed time to assume bewteen frames
 */
auto run(Window& window, const auto& update, const auto& draw, u64 substeps, f64 dt)
{
    while (window.is_open())
    {
        for (auto i : loop::end(substeps))
        {
            // TODO is this correct dt?
            // https://youtu.be/yGhfUcPjXuE?t=333 use diff bw n-1 and n-2
            update(dt / substeps);
            // std::ignore = i;
        }
        draw();
        window.display();
    }
}

// TODO deprecate
auto zoom_pan(Window& window, f64 zoom_factor = 0.1, f64 pan_factor = 1.0)
{
    if (window.mouse.left)
    {
        window.view.pos += pan_factor * (window.mouse.pos - window.mouse.old_pos);
    }

    const auto scale = 1.0 + zoom_factor * window.mouse.scroll_amount;
    window.view.scale *= Vec2::combine(scale);
    window.view.pos = window.mouse.pos + scale * pan_factor * (window.view.pos - window.mouse.pos);
}

auto zoom_pan(Window& window,
              const std::invocable auto& zoom_condition,
              const std::invocable auto& pan_condition,
              f64 zoom_factor = 0.1,
              f64 pan_factor  = 1.0)
{
    if (pan_condition())
    {
        window.view.pos += pan_factor * (window.mouse.pos - window.mouse.old_pos);
    }
    if (zoom_condition())
    {
        const auto scale = 1.0 + zoom_factor * window.mouse.scroll_amount;
        window.view.scale *= Vec2::combine(scale);
        window.view.pos =
            window.mouse.pos + scale * pan_factor * (window.view.pos - window.mouse.pos);
    }
}
} // namespace sm
