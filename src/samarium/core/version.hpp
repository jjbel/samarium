/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "fmt/format.h"

#include "types.hpp"

namespace sm
{
struct Version
{
    u8 major{1};
    u8 minor{0};
    u8 patch{0};
};

static constexpr auto version = Version{};
} // namespace sm

template <> class fmt::formatter<sm::Version>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const sm::Version& p, FormatContext& ctx)
    {
        return format_to(ctx.out(), "samarium version {}.{}.{}", p.major, p.minor,
                         p.patch);
    }
};
