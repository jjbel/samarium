#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{.dims{1200, 600}}};

    auto draw = [&]
    {
        draw::background("#07090b"_c);

        constexpr auto a = 0.5;
        for (auto pos : std::vector<Vector2>{{0, 0}, {a, a}, {-a, a}, {a, -a}, {-a, -a}})
        {
            draw::circle(window, Circle{pos, 0.2}, "#ff0000"_c, 4);
            // draw::circle(window, Circle{pos, 0.2}, "#ff0000"_c, Transform{}, 4);
        }
    };

    auto count = 0;
    auto watch = Stopwatch{};

    while (window.is_open())
    {
        // print(window.dims, window.squash);


        // window.pan();
        // window.zoom_to_cursor();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        count++;
        draw();
        window.display();
    }
}

// TODO resizing window: see glViewport. draw 4 circles at GL corners and see if they remain at
// corners
