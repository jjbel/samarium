#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "turtle.hpp"


using namespace sm;
using namespace sm::literals;

static auto turtle = Turtle{};

int main()
{
    auto window       = Window{{{500, 500}, "Turtle Sim", false}};
    window.view.scale = Vector2::combine(1.0 / 250.0);

    const auto update = [&]
    {
        draw::background(colors::white);
        turtle.draw(window);

        // turtle.pos.x += 0.001;
        // turtle.pos.y -= 0.001;
        turtle.angle += 0.0005;
        turtle.forward(0.03);
    };

    run(window, update);
}
