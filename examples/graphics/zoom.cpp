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

        constexpr auto a = 0.2;
        for (auto pos : std::vector<Vector2>{{0, 0}, {a, a}, {-a, a}, {a, -a} /* , {-a, -a} */})
        {
            draw::circle(window, Circle{pos, 0.08}, "#ff0000"_c, 64);
        }
    };

    // TODO on the 2nd frame, mouse delta is non-zero
    // can fix by calling display twice, or maybe getInput twice?
    // window.display();
    // window.display();
    auto count = 0;
    auto watch = Stopwatch{};

    while (window.is_open())
    {
        window.pan();
        window.zoom_to_cursor();
        // window.zoom_to_origin();

        // TODO sleep or stopwatch not accurate? 4ms makes it go to 16ms
        // std::this_thread::sleep_for(std::chrono::milliseconds(4));

        count++;
        draw();
        window.display();
    }

    const auto fps = count / watch.seconds();
    const auto ms  = 1000.0 / fps;
    print(fmt::format("{: 6.2f}fps, {: 5.2f}ms", fps, ms));
}

// TODO dont pass glm mat4x4 to shader at all (can use the overloads in future)
// include "glm/gtx/string_cast.hpp"     // TODO remove for to_string
// TODO shd include <glm/gtc/type_ptr.hpp>

// TODO header only makes it faster?
// TODO resizing window: see glViewport. draw 4 circles at GL corners and see if they remain at
// corners
