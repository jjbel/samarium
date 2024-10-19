#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"


using namespace sm;

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};
    auto plot   = Plot{};

    const auto grid_dims = Dimensions{2, 2};

    auto rng = RandomGenerator{};

    plot.traces["x"] = {colors::red};
    plot.traces["y"] = {colors::green};
    plot.traces["z"] = {colors::blue};

    print("Hello");

    auto frame_counter = 0;
    const auto draw    = [&]
    {
        plot.transform.pos.x = -0.5;

        const auto boxes = subdivide_box(window.world_box(), grid_dims, 0.97);
        plot.box         = boxes[{0, 0}];

        plot.add("x", Vector2{frame_counter / 1000.0,
                              noise::perlin_1d(frame_counter / 1000.0, {100.0}) - 0.5});

        plot.add("y", Vector2{frame_counter / 1000.0,
                              noise::perlin_1d(frame_counter / 1000.0 + 100.0, {100.0}) - 0.5});

        plot.add("z", Vector2{frame_counter / 1000.0,
                              noise::perlin_1d(frame_counter / 1000.0 + 200.0, {100.0}) - 0.5});

        draw::background(colors::black);
        plot.draw(window);

        // print(plot.traces["x"].points, plot.traces["y"].points, plot.traces["z"].points);

        frame_counter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    };
    run(window, draw);
}
