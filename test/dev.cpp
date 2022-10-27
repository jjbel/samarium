#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto window = Window{{1280, 720}};
    auto text   = expect(
        draw::Text::make("/usr/share/fonts/TTF/Fira Code Regular Nerd Font Complete Mono.ttf"));

    while (window.is_open())
    {
        draw::background("#141414"_c);
        draw::circle(window, {.centre = {}, .radius = .3}, {.fill_color = colors::red});
        // text(window, "Bezier Curves", {200.0F, 300.0F}, 1.0F, colors::ivory);
        text(window, "libsamarium", {}, 1.0F, colors::ivory);
        window.display();
    }
    print("DoNnnnn");
}
