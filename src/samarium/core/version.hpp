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
struct version_t
{
    u8 major{1};
    u8 minor{0};
    u8 patch{0};
};

static constexpr auto version = version_t{};
} // namespace sm

template <> class fmt::formatter<sm::version_t>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const sm::version_t& p, FormatContext& ctx)
    {
        return format_to(ctx.out(), "samarium version {}.{}.{}", p.major, p.minor,
                         p.patch);
    }
};
