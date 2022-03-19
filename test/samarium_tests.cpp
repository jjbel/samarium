/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/colors.hpp"
#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/gui/Window.hpp"
#include "../src/samarium/gui/sfml.hpp"
#include "../src/samarium/math/Dual.hpp"
#include "../src/samarium/physics/collision.hpp"
#include "../src/samarium/physics/fluid/Fluid.hpp"
#include "../src/samarium/util/file.hpp"

using sm::print;

void App();

int main() { App(); }

void App()
{
    // auto rn     = sm::Renderer{};
    // auto window = sm::Window{{.dims = rn.image.dims, .name = "Samarium", .framerate = 64}};

    // const auto update = [&](auto /* delta */) {};


    // const auto draw = [&]
    // {
    //     rn.fill(sm::Color{16, 18, 20});


    //     rn.render();
    // };

    // window.run(rn, update, draw, 40, 700);

    const auto f = sm::Fluid{};
    fmt::print(fg(fmt::color::light_green) | fmt::emphasis::bold, "Done\n");
}
