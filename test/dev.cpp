/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app = App{{.dims = dims720}};

    const auto noise = util::PerlinNoise{};

    const auto draw = [&]
    {
        sf::Font font;
        if (!font.loadFromFile("/home/jb/Downloads/Noto_Serif_JP/NotoSerifJP-ExtraLight.otf"))
        {
            throw std::exception();
        }
        sf::Text text;

        // select the font
        text.setFont(font); // font is a sf::Font

        // set the string to display
        text.setString("Hello world");

        // set the character size
        // text.setCharacterSize(48); // in pixels, not points!

        app.sf_render_window.draw(text);

        app.draw_world_space(
            [&noise](Vector2 pos)
            {
                pos /= 10.0;
                return Color::from_grayscale(u8(noise(pos.x, pos.y) * 255));
            });
    };
    app.run(draw);
}
