/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <complex>

#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/samarium.hpp"

using namespace sm;

auto to_complex(Vector2 vec) { return std::complex{vec.x, vec.y}; }


auto mandelbrot_iter(Vector2 pos, f64 threshold, u64 max_iterations) -> std::optional<u64>
{
    auto z                 = std::complex<f64>{};
    const auto pos_complex = to_complex(pos);

    for (auto i : range(max_iterations))
    {
        z = z * z + pos_complex;
        if (std::abs(z) > threshold) return std::optional<u64>{i};
    }

    return std::nullopt;
};

auto mandelbrot(Vector2 pos, auto&& gradient, f64 threshold = 12.0, u64 max_iterations = 16)
{
    if (const auto iter = mandelbrot_iter(pos, threshold, max_iterations); iter.has_value())
    {
        return gradient(interp::map_range_clamp(
            static_cast<f64>(*iter), {0.0, static_cast<f64>(max_iterations)}, {0.0, 1.0}));
    }
    else
    {
        return Color::from_hex("#101013");
    }
}

int main()
{
    auto rn = Renderer{Image{dims720}};
    rn.transform.pos += Vector2{.x = 400};
    rn.transform.scale *= 40;

    const auto drawer = [](auto pos) { return mandelbrot(pos, sm::gradients::heat); };

    auto window = Window{{.dims = rn.image.dims}};

    auto w = util::Stopwatch{};

    while (window.is_open())
    {
        window.get_input();
        print(rn.transform.apply_inverse(window.mouse.pos.now));
        rn.draw(drawer);
        window.draw(rn.image);
        window.display();
        w.print();
        w.reset();
    }
}
