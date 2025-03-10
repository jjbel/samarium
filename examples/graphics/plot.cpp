#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window          = Window{{.dims = dims720}};
    auto plot            = Plot("CascadiaCode.ttf");
    const auto grid_dims = Dimensions{2, 2};
    auto rng             = RandomGenerator{};
    plot.traces["x"]     = {"#ff0f0f"_c, 0.008F};
    plot.traces["y"]     = {"#05ff00"_c, 0.008F};
    plot.traces["z"]     = {"#004dff"_c, 0.008F};
    plot.title.text      = "Acceleration";
    auto frame_counter   = 0;
    const auto draw      = [&]
    {
        const auto boxes = subdivide_box(window.world_box(), grid_dims, 0.97);
        plot.box         = boxes[{0, 1}];
        plot.add("x", Vec2{frame_counter / 1000.0,
                           noise::perlin_1d(frame_counter / 1000.0, {100.0}) - 0.5});
        plot.add("y", Vec2{frame_counter / 1000.0,
                           noise::perlin_1d(frame_counter / 1000.0 + 100.0, {100.0}) - 0.5});
        plot.add("z", Vec2{frame_counter / 1000.0,
                           noise::perlin_1d(frame_counter / 1000.0 + 200.0, {100.0}) - 0.5});

        draw::background(colors::black);
        plot.draw(window);
        frame_counter++;
    };
    run(window, draw);
}
