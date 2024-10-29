#include "samarium/samarium.hpp"

using namespace sm;

auto main() -> i32
{
    auto window          = Window{{.dims = dims720}};
    auto frame_counter   = 0;
    const auto draw      = [&]
    {
        draw::background(colors::black);
        frame_counter++;
    };
    run(window, draw);
}
