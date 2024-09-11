#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"


using namespace sm;

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};

    // TODO remove this reference for copypasteability
    auto& mouse = window.mouse;

    auto draw = [&]
    {
        draw::background(colors::midnightblue);
        constexpr auto a = 0.2;
        for (auto pos : std::vector<Vector2>{{0, 0}, {a, a}, {-a, a}, {a, -a}, {-a, -a}})
        {
            draw::circle(window, Circle{pos, 0.08}, colors::red, 64);
        }
    };

    const auto scroll_factor = 1.1;
    // auto scroll = 1.0;

    // ---------

    window.view = Transform{};
    // ---------
    auto count = 0;

    // TODO on the 2nd frame, mouse delta is non-zero
    // can fix by calling display twice, or maybe getInput twice?
    // window.display();
    // window.display();
    // window.display();

    while (window.is_open())
    {
        const auto delta = mouse.pos - mouse.old_pos;
        window.view.scale *= std::pow(scroll_factor, mouse.scroll_amount);
        if (window.mouse.left)
        {
            window.view.pos += window.view_px2gl()(mouse.pos) - window.view_px2gl()(mouse.old_pos);
        }

        draw();
        count++;
        window.display();
    }
}
