#include "../src/samarium/samarium.hpp"

int main()
{
    using namespace sm;
    print("Hello there Ccache");
    file::export_tga(Image{}, "Test.tga");
}
