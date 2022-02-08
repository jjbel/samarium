#include "samarium/Colors.hpp"
#include "samarium/Renderer.hpp"
#include "samarium/file.hpp"
#include "samarium/random.hpp"
#include "samarium/interp.hpp"
#include <execution>
#include <ranges>

using sm::util::print;

int main()
{
    auto im = sm::Image{ sm::dimsFHD, sm::colors::darkslateblue };
    auto fn = [dims = im.dims](sm::Indices i)
    {
        return sm::colors::bisque.with_alpha(
            sm::math::map_range(std::sin(i.x / 20.), -1., 1., 0, 255));
    };

    auto w = sm::util::Stopwatch{};

    for (size_t y = 0; y < im.dims.y; y++)
        for (size_t x = 0; x < im.dims.x; x++) im[{ x, y }].add_alpha_over(fn({ x, y }));


    w.print();

    sm::file::export_to(im, "temp1.tga", true);
}
