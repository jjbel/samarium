#include "samarium/Colors.hpp"
#include "samarium/Renderer.hpp"
#include "samarium/Window.hpp"
#include "samarium/file.hpp"
#include "samarium/interp.hpp"
#include "samarium/random.hpp"
#include <execution>
#include <ranges>


using sm::util::print;
namespace si = sm::interp;

int main()
{
    using namespace sm::literals;
    auto image  = sm::Image{ sm::dimsFHD, "#1202e9"_c };
    // auto mapper = sm::interp::make_clamped_mapper(sm::Extents{ 0ul, image.dims.x },
    //                                               sm::Extents{ 0, 255 });
    // auto fn     = [dims = image.dims, mapper](sm::Indices i)
    // {
    //     return sm::Color("#f64f59").with_alpha(
    //         sm::interp::clamp((50 - (i - sm::Indices{ 500, 500 }).length()) / 2. + 1,
    //                           { 0., 1. }) *
    //         255);
    // };

    // print(si::map_range(.4, sm::Extents{0., 1.}, sm::Extents{-1., 1.}));

    // auto w = sm::util::Stopwatch{};

    // for (size_t y = 0; y < image.dims.y; y++)
    //     for (size_t x = 0; x < image.dims.x; x++)
    //         image[{ x, y }].add_alpha_over(fn({ x, y }));


    // w.print();
    print("#1202e9"_c);

    sm::file::export_to(image, "temp1.tga", true);
    // auto win = sm::Window{ image.dims, "Hello Moto" };

    // w.reset();
    // while (win.is_open())
    // {
    //     win.get_input();
    //     image[{ 10, 10 }] = sm::colors::black;
    //     image[{ 10, 11 }] = sm::colors::black;
    //     image[{ 10, 12 }] = sm::colors::black;
    //     image[{ 11, 12 }] = sm::colors::black;
    //     image[{ 12, 12 }] = sm::colors::black;
    //     win.draw(image);
    //     win.display();
    // }
    // auto d = w.time() / 1.e3;
    // fmt::print("Frames: {}, time: {}, fr: {}, time: {}\n", win.frame, d, win.frame / d,
    //            1000.0 / (win.frame / d));
    sm::Extents<int> e;
    print(e.min, e.max);
    print(sm::Color{ .r = 12, .a = 1 });
}
