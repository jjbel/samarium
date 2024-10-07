#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto square(Vector2 centre, f64 size)
{
    return std::vector{centre + Vector2{size, size}, centre + Vector2{size, -size},
                       centre + Vector2{-size, -size}, centre + Vector2{-size, size}};
}

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};
    window.camera.scale /= 6.0;

    auto in_pts  = points_to_f32(square({}, 2.0));
    auto out_pts = draw::make_polyline(in_pts, 1.0F);
    print(out_pts);

    auto left_old = false;

    auto draw = [&]
    {
        draw::background("#07090b"_c);

        draw::polyline(window, in_pts, 0.3F, "#0000ff"_c);

        // TODO gives flaky:

        // TODO sometimes flips to concave polys
        // if (window.mouse.left && !left_old)
        if (window.mouse.pos != window.mouse.old_pos)
        {
            in_pts.push_back(window.pixel2world()(window.mouse.pos).cast<f32>());
            print(in_pts);
        }
        left_old = window.mouse.left;

        // for (auto pos : in_pts)
        // {
        //     draw::circle(window, Circle{pos.cast<f64>(), 0.2}, "#ff0000"_c, 64);
        // }
        // for (auto pos : out_pts)
        // {
        //     draw::circle(window, Circle{pos.cast<f64>(), 0.2}, "#0000ff"_c, 64);
        // }
    };

    auto count = 0;
    auto watch = Stopwatch{};
    run(window,
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            count++;
            draw();
        });
    // print(count / watch.time());
}
