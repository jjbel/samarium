#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"


using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{.dims{1200, 600}}};

    // TODO remove this reference for copypasteability
    auto& mouse = window.mouse;

    auto f = [&](Vector2 p)
    {
        const auto pos = p.cast<f32>();
        // const auto points = math::regular_polygon_points<f32>(64, Circle{pos, 0.08});
        const auto a = 0.06F;
        // TODO polygon: specify pts in clock order
        auto points = std::vector<Vector2f>({Vector2f{-a, -a} + pos, Vector2f{-a, a} + pos,
                                             Vector2f{a, a} + pos, Vector2f{a, -a} + pos});

        const auto t = window.world2gl();
        // for (auto& pt : points) { pt = t(pt.cast<f64>()).cast<f32>(); }

        // TODO to_string is very slow, shd use fmt
        // print(glm::to_string(bt.as_matrix()));

        draw::polygon(window, points, "#ff0000"_c, window.world2gl());
    };

    // TODO shd include <glm/gtc/type_ptr.hpp>

    auto draw = [&]
    {
        draw::background("#07090b"_c);

        constexpr auto a = 0.2;
        for (auto pos : std::vector<Vector2>{{0, 0}, {a, a}, {-a, a}, {a, -a} /* , {-a, -a} */})
        {
            f(pos);
        }
    };

    const auto scroll_factor = 1.2;
    auto count               = 0;

    // TODO on the 2nd frame, mouse delta is non-zero
    // can fix by calling display twice, or maybe getInput twice?
    // window.display();
    // window.display();
    // window.display();

    // TODO add to and from camera

    // see zoom pan in run.hpp


    while (window.is_open())
    {
        const auto pos     = window.pixel2world()(mouse.pos);
        const auto old_pos = window.pixel2world()(mouse.old_pos);

        const auto scale = std::pow(scroll_factor, mouse.scroll_amount);
        window.camera.pos = pos + scale * (window.camera.pos - pos);
        window.camera.scale *= scale;
        if (window.mouse.left) { window.camera.pos += pos - old_pos; }

        // print(/* mouse.pos,  */ window.camera.inverse()(window.pixel2view()(mouse.pos)),
        // window.camera);

        // window.camera.pos = {0, 0.25};
        // print((window.camera({0, 0})), window.world2gl()({0, 0}));
        // print(window.world2gl()({0, 0}));
        // print(window.world2gl());
        // print(window.squash);

        draw();
        count++;
        window.display();
    }
}

// TODO dont pass glm mat4x4 to shader at all (can use the overloads in future)
// include "glm/gtx/string_cast.hpp"     // TODO remove for to_string 