#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

struct ParametricFunction
{
    f64 a{1};
    f64 b{1};
    f64 c{1};
    f64 d{1};
    f64 scale{1};

    constexpr auto operator()(f64 t) const
    {
        return scale * Vector2{std::cos(a * t) - math::power<3>(std::cos(b * t)),
                               std::sin(c * t) - math::power<3>(std::sin(d * t))};
    }
};

auto main() -> i32
{
    auto window      = Window{{{1800, 900}}};
    auto fn          = ParametricFunction{1, 40, 1, 40, 5};
    auto time        = 0.0;
    const auto speed = 1e-3;
    auto trail       = Trail{20000};

    const auto update = [&]
    {
        draw::background("#ffebe2"_c);
        draw::grid_lines(
            window,
            {.spacing = 1, .color = "#0a126a"_c.with_multiplied_alpha(0.15), .thickness = 0.03F});
        const auto pos = fn(time);
        trail.push_back(pos);
        draw::trail(window, trail, "#0a126a"_c.with_multiplied_alpha(0.8), 2);
        time += speed;
    };

    run(window, update);
}
