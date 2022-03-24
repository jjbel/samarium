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
#include "../src/samarium/util/format.hpp"

void App();

int main() { App(); }

void App()
{
    using namespace sm;

    const auto dims = Dimensions{512, 512};
    const auto bbox = BoundingBox<i64>{{}, dims.as<i64>()};
    auto fluid      = Fluid{};

    auto window = Window{{dims}};

    size_t frame{};
    while (window.is_open() && ++frame <= 200)
    {
        fmt::print(stderr, "\n{}: ", window.frame_counter);
        window.get_input();

        if (const auto pos = window.mouse.pos->as<i64>(); window.mouse.left && bbox.contains(pos))
        {
            print("Click!", pos);
            // image[pos.as<u64>()] = colors::white;
        }

        // window.draw(fluid.to_image());
        window.display();
    }
}
