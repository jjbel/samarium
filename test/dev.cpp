#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"
#include "turtle.hpp"

using namespace sm;

static auto turtle = Turtle{};

int main()
{
    auto window       = Window{{{500, 500}, "Turtle Sim", false}};
    window.view.scale = Vector2::combine(1.0 / 250.0);

    const auto update = [&]
    {
        draw::background(colors::white);
        turtle.draw(window);
        turtle.angle += 0.0005;
        turtle.forward(0.06);
    };

    run(window, update);
}
