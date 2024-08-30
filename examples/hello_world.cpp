#include "samarium/samarium.hpp"

using namespace sm;

int main()
{
    print("Hello World!");
    print("A Vector2:", Vector2{.x = 5, .y = -3});
    print("A Color:  ", Color{.r = 5, .g = 200, .b = 255});
}
