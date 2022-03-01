/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

// #include

template <typename T> struct Dual
{
    T now{};
    T prev{now};

    constexpr auto& operator*() { return now; }

    constexpr auto operator->() { return &now; }

    constexpr auto update(auto&&... args)
    {
        now.update(args...);
        prev = now;
    }
};
