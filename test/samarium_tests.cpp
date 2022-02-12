#include "samarium/Colors.hpp"
#include "samarium/Renderer.hpp"
#include "samarium/Window.hpp"
#include "samarium/file.hpp"
#include "samarium/interp.hpp"
#include "samarium/random.hpp"
#include <execution>
#include <functional>
#include <ranges>


using sm::util::print;
namespace si = sm::interp;

int main()
{
    // using namespace sm::literals;
    // auto image = sm::Image{ sm::dimsHD, "#10101A"_c };

    // const auto mapper = sm::interp::make_clamped_mapper<size_t, uint8_t>(
    //     sm::Extents{ 0ul, image.dims.x }, sm::Extents{ 0ul, 255ul });

    // const auto draw_circle = [dims = image.dims, mapper](sm::Vector2 i,
    //                                                      sm::Vector2 centre,
    //                                                      double radius, double aa)
    // {
    //     return sm::Color(255, 50, 50,100)
    //         .with_multiplied_alpha(
    //             sm::interp::clamp((radius - (i - centre).length()) / aa + 1, { 0., 1. }));
    // };

    // auto w = sm::util::Stopwatch{};

    // std::vector<std::function<sm::Color(const sm::Vector2&)>> buffer;
    // for (size_t i = 0; i < 10; i++)
    // {
    //     buffer.emplace_back(
    //         [draw_circle, i](sm::Vector2 coords) {
    //             return draw_circle(coords, sm::Vector2{ .8, .6 } * (30. * i), 60., 2.);
    //         });
    // }


    // const sm::Color* beg = image.begin();
    // std::for_each(std::execution::par, image.begin(), image.end(),
    //               [=, dims = image.dims](auto& pixel)
    //               {
    //                   const auto coords = sm::convert_1d_to_2d(dims, &pixel - beg);
    //                   for (const auto& fn : buffer) pixel.add_alpha_over(fn(coords));
    //               });


    // w.print();

    // sm::file::export_to(image, "temp1.tga", true);
    // for (auto win = sm::Window{ image.dims }; win;)
    // {
    //     win.get_input();

    //     win.draw(image);
    //     win.display();
    // }
}
