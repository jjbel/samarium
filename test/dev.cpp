#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"


using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};
    auto plot   = Plot("CascadiaCode.ttf");

    const auto grid_dims = Dimensions{2, 2};

    auto rng = RandomGenerator{};

    plot.traces["x"] = {"#ff0f0f"_c};
    plot.traces["y"] = {"#05ff00"_c};
    plot.traces["z"] = {"#004dff"_c};
    plot.title.text = "Acceleration";

    // auto text = expect(draw::Text::make("D:\\fonts\\Inter\\static\\Inter_24pt-Regular.ttf"));
    // auto text = expect(draw::Text::make("D:\\fonts\\arial.ttf"));
    auto text = expect(draw::Text::make("CascadiaCode.ttf"));

    auto frame_counter = 0;

    const auto str = "This is sample text. 0123456789";
    auto box       = text.bounding_box(str, 0.2, {PlacementX::Middle, PlacementY::Middle});

    const auto draw = [&]
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

        // draw::bounding_box(window, box, colors::red, 0.01F);
        // text(window, str, {}, 0.2);

        // print(plot.traces["x"].points, plot.traces["y"].points, plot.traces["z"].points);

        window.pan();
        window.zoom_to_cursor();

        frame_counter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    };
    run(window, draw);
}
