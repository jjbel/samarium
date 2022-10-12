/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <iostream>

#include "samarium/util/format.hpp"

auto operator<<(std::ostream& os, const sm::Version& a) -> std::ostream&;

auto operator<<(std::ostream& os, const sm::Color& a) -> std::ostream&;

template <sm::concepts::Number T>
auto operator<<(std::ostream& os, const sm::BoundingBox<T>& a) -> std::ostream&
{
    os << fmt::to_string(a);
    return os;
}

template <sm::concepts::Number T>
auto operator<<(std::ostream& os, const sm::Vector2_t<T>& a) -> std::ostream&
{
    os << fmt::to_string(a);
    return os;
}


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_OSTREAM_IMPL)

#include "fmt/format.h"

std::ostream& operator<<(std::ostream& os, const sm::Version& a)
{
    os << fmt::to_string(a);
    return os;
}

std::ostream& operator<<(std::ostream& os, const sm::Color& a)
{
    os << fmt::to_string(a);
    return os;
}

#endif
