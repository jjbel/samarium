#include "samarium/Colors.hpp"
#include "samarium/Renderer.hpp"
#include "samarium/Window.hpp"
#include "samarium/file.hpp"
#include "samarium/interp.hpp"
#include "samarium/random.hpp"
#include <execution>
#include <ranges>


using sm::util::print;

int main()
{
    using namespace sm::literals;
    auto image  = sm::Image{ { 100, 100 }, "#12c2e9"_c };
    auto mapper = sm::interp::make_clamped_mapper(sm::Extents{ 0ul, image.dims.x },
                                                  sm::Extents{ 0, 255 });
    auto fn     = [dims = image.dims, mapper](sm::Indices i)
    {
        return sm::Color("#f64f59").with_alpha(
            sm::interp::clamp((50 - (i - sm::Indices{ 500, 500 }).length()) / 2. + 1,
                              { 0., 1. }) *
            255);
    };

    auto w = sm::util::Stopwatch{};

    for (size_t y = 0; y < image.dims.y; y++)
        for (size_t x = 0; x < image.dims.x; x++)
            image[{ x, y }].add_alpha_over(fn({ x, y }));


    w.print();


    // sm::file::export_to(im, "temp1.tga", true);
    auto win = sm::Window{ image, "Hello Moto", 60 };

    w.reset();
    while (win.is_open())
    {
        win.get_input();
        image[{ 10, 10 }] = sm::colors::black;
        // window.clear();
        win.display();
    }
    auto d = w.time() / 1.e3;
    fmt::print("Frames: {}, time: {}, fr: {}\n", win.frame, d, win.frame / d);
    print((sm::Vector2(1, 3) - sm::Vector2(3, -6)).length());
}
