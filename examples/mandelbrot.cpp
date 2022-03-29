/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <complex>

#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/math/complex.hpp"
#include "../src/samarium/samarium.hpp"

using namespace sm;

namespace mandelbrot
{
auto get_iter(Vector2 pos, f64 threshold, u64 max_iterations) -> std::optional<u64>
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

auto colorise(Vector2 pos, auto&& gradient, f64 threshold = 42.0, u64 max_iterations = 40)
{
    if (const auto iter = get_iter(pos, threshold, max_iterations); iter.has_value())
    {
        return gradient(interp::map_range_clamp(
            static_cast<f64>(*iter), {0.0, static_cast<f64>(max_iterations)}, {0.0, 1.0}));
    }
    else
    {
        return Color{9, 5, 26};
    }
}

static auto iters = 40UL;

const auto draw = [](auto pos) { return colorise(pos, gradients::magma, 42.0, iters); };
} // namespace mandelbrot

int main()
{
    auto rn = Renderer{Image{dims720}};
    rn.transform.pos += Vector2{.x = 400};
    rn.transform.scale *= 40;

    auto window = Window{{.dims = rn.image.dims}};

    while (window.is_open())
    {
        window.get_input();

        if (window.mouse.left) { rn.transform.pos += window.mouse.pos.now - window.mouse.pos.prev; }

        const auto scale = 1.0 + 0.1 * window.mouse.scroll_amount;
        rn.transform.scale *= Vector2::combine(scale);
        const auto pos    = (window.mouse.pos.now);
        rn.transform.pos  = pos + scale * (rn.transform.pos - pos);
        mandelbrot::iters = static_cast<u64>(3 * std::log(rn.transform.scale.x) + 9);

        rn.draw(mandelbrot::draw);
        window.draw(rn.image);
        window.display();
    }
}
