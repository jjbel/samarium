/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/samarium.hpp"
#include "samarium/graphics/Image.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/util/Stopwatch.hpp"
#include "samarium/util/random.hpp"

using namespace sm;
using namespace sm::literals;

struct Params
{
    Dimensions dims = dims720;
};

struct NoitaSandSim
{
    static constexpr auto upscale_factor = 2UL;

    enum class Element : u8
    {
        Sand,
        Water,
        Empty
    };

    struct particle_t
    {
        Vector2 vel{};
        Element element;
    };

    Grid<particle_t> particles;
    Image image;

    explicit NoitaSandSim(Dimensions dims) : particles{dims / upscale_factor}, image{dims} {}

    [[nodiscard]] static constexpr auto get_color(Element element) noexcept
    {
        using enum Element;

        switch (element)
        {
        case Sand: return "#db9b3b"_c; break;
        case Water: return "#4060ff"_c; break;
        default: return "#1a1b26"_c; break;
        }
    }
};

int main()
{
    using enum NoitaSandSim::Element;

    const auto params = Params{};
    auto sim          = NoitaSandSim{params.dims};
    auto app          = App{{.dims = params.dims}};

    auto active_element = Sand;

    auto update = [&](auto /* delta */)
    {
        const auto mouse_pos = (app.mouse.pos.now).as<u64>();
        // sim.particles[mouse_pos / NoitaSandSim::upscale_factor].element = active_element;
        // print(mouse_pos);
    };

    auto watch = Stopwatch{};

    auto draw = [&]
    {
        app.draw([](auto coords) { return colors::pink; }, {{-0.2, -0.2}, {0.2, 0.2}});

        app.draw(Circle{.centre{10, 10}, .radius = 3}, colors::red);

        watch.print();
        watch.reset();
    };

    app.run(update, draw);
}
