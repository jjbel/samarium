#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{1280, 720}};
    auto text   = expect(
        draw::Text::make("/usr/share/fonts/TTF/Fira Code Regular Nerd Font Complete Mono.ttf"));
    auto watch = Stopwatch{};

    while (window.is_open())
    {
        watch.reset();
        draw::background("#141414"_c);
        draw::circle(window, {.centre = {}, .radius = .3}, {.fill_color = colors::red});
        text(window, "Bezier Curves", {0.2000F, 0.3000F}, 1.0F, colors::ivory);
        text(window, fmt::format("{:3.2} ms", watch.seconds() * 1000), {}, 1.0F, colors::ivory);
        window.display();
    }
}
