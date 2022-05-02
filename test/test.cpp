/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

namespace sm
{
struct Mesh
{
    using Vertex = Vector2;

    struct Edge
    {
        u64 v1{};
        u64 v2{};
    };

    std::vector<Vertex> vertices{};
    std::vector<Edge> edges{};
};
} // namespace sm

int main()
{
    auto app  = App{{.dims = dims720}};
    auto mesh = Mesh{};
}
