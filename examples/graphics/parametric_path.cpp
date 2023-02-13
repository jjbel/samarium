#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

// Lissajous Figure
struct ParametricFunction
{
    f64 f_x{10};
    f64 f_y{4};
    f64 delta{0.5};

    f64 scale{10};
    f64 speed = 0.05;

    constexpr auto operator()(f64 t) const
    {
        return scale * Vector2{std::sin(f_x * t + math::pi * delta), std::sin(f_y * t)};
    }
};

// struct ParametricFunction
// {
//     f64 a{1};
//     f64 b{40};
//     f64 c{1};
//     f64 d{40};
//     f64 scale{5};
//     f64 speed{1e-3};

//     constexpr auto operator()(f64 t) const
//     {
//         return scale * Vector2{std::cos(a * t) - math::power<3>(std::cos(b * t)),
//                                std::sin(c * t) - math::power<3>(std::sin(d * t))};
//     }
// };

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};
    auto fn     = ParametricFunction{};
    auto time   = 0.0;
    auto trail  = Trail{20000};

    const auto update = [&]
    {
        draw::background("#ffebe2"_c);
        draw::grid_lines(
            window,
            {.spacing = 1, .color = "#0a126a"_c.with_multiplied_alpha(0.15), .thickness = 0.03F});
        const auto pos = fn(time);
        trail.push_back(pos);
        draw::trail(window, trail, "#0a126a"_c.with_multiplied_alpha(0.8), 2);
        time += fn.speed;
    };

    run(window, update);
}
