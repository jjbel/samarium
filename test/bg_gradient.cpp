#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"


using namespace sm;

auto main() -> i32
{
    auto window        = Window{{.dims = dims720}};
    auto frame_counter = 0;
    while (window.is_open())
    {
        print("a");
        // draw::background({});
        draw::background(window, gradients::viridis);
        print("b");
        frame_counter++;
        window.display();
        print("c");
        print();
    };
}
