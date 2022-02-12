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
    auto image        = sm::Image{ sm::dimsHD, "#10101A"_c };
    const auto mapper = sm::interp::make_clamped_mapper(sm::Extents{ 0ul, image.dims.x },
                                                        sm::Extents{ 0ul, 255ul });
    const auto fn     = [dims = image.dims, mapper](sm::Indices i)
    {
        return sm::colors::orangered.with_alpha(
            sm::interp::clamp((50 - (i - sm::Indices{ 500, 500 }).length()) / 2. + 1,
                              { 0., 1. }) *
            255);
    };

    // print(si::map_range(.4, sm::Extents{0., 1.}, sm::Extents{-1., 1.}));


    auto w = sm::util::Stopwatch{};

    // for (size_t y = 0; y < image.dims.y; y++)
    //     for (size_t x = 0; x < image.dims.x; x++)
    //         image[{ x, y }].add_alpha_over(fn({ x, y }));
    sm::Color* beg = image.begin();
    std::for_each(image.begin(), image.end(),
                  [beg](auto& pixel) { auto [x, y] = sm::convert_1d_to_2d(beg - &pixel); });


    w.print();

    // sm::file::export_to(image, "temp1.tga", true);
    for (auto win = sm::Window{ image.dims }; win;)
    {
        win.get_input();

        win.draw(image);
        win.display();
    }
}
