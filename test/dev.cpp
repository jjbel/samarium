#include "samarium/gl/draw/grid.hpp"
#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};

    const auto update = [&]
    {
        draw::background("#12121a"_c);
        draw::grid_lines(window);
        draw::circle(window, {{0, 0}, 0.6}, {colors::crimson});
        print(window.dims, window.aspect_ratio(), window.view.scale.y / window.view.scale.x);
    };

    run(window, update);
}
