/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"
#include <ranges>

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app  = App{{.dims = dims720}};
    auto mesh = Mesh{};

    // auto v = std::views::transform(mesh.vertices, [](const auto& v) { return v.x; });
}
