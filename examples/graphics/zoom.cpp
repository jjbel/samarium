#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};

    auto draw = [&]
    {
        draw::background("#07090b"_c);

        constexpr auto a = 0.2;
        for (auto pos : std::vector<Vector2>{{0, 0}, {a, a}, {-a, a}, {a, -a}})
        {
            draw::circle(window, Circle{pos, 0.08}, "#ff0000"_c, 64);
        }
    };

    run(window,
        [&]
        {
            window.pan();
            window.zoom_to_cursor();
            // window.zoom_to_origin();

            draw();
        });
}
