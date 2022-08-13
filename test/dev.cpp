/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/action/remove_if.hpp"
#include "range/v3/algorithm/contains.hpp"
#include "range/v3/algorithm/remove_copy_if.hpp"
#include "range/v3/view/take.hpp"

using namespace sm;
using namespace sm::literals;

using complex = std::complex<f64>;

constexpr auto integration_steps = 256UL;


auto raise_to_power(complex x) { return std::pow(math::e, math::two_pi_i * x); }

auto sample_at(const std::vector<complex>& points, f64 factor)
{
    const auto floor = static_cast<u64>(std::floor(factor * static_cast<f64>(points.size())));
    if (floor == 0) { return points[0]; }
    return interp::lerp<complex>(factor - static_cast<f64>(floor),
                                 {points[floor], points[floor - 1]});
}

auto map_index(i32 i)
{
    if (i % 2 == 0) { return i / 2; }
    else { return -(i + 1) / 2; }
}

auto indices(u64 resolution)
{
    return ranges::views::iota(0) | ranges::views::transform(map_index) |
           ranges::views::take(resolution);
}

struct FourierState
{
    std::vector<complex> coefficients{};

    auto refresh(const std::vector<complex>& points, u64 resolution)
    {
        coefficients.clear();
        coefficients.reserve(resolution);

        for (auto j : indices(resolution))
        {
            const auto function = [&](f64 factor)
            { return raise_to_power(-factor * f64(j)) * sample_at(points, factor); };

            const auto coefficient =
                math::integral<complex, f64>(function, 0.0, 1.0, integration_steps);
            coefficients.push_back(coefficient);
        }

        for (auto i : coefficients) { print(i); }
    }

    auto draw(App& app, f64 time)
    {
        auto sum = complex{};
        for (auto [index, coeff] : ranges::views::zip(indices(coefficients.size()), coefficients))
        {
            auto current = coeff * raise_to_power(f64(index) * time); // base vector

            app.draw_line_segment({from_complex(sum), from_complex(sum + current)}, colors::white,
                                  0.08);
            sum += current;
        }
    }
};

enum class Mode
{
    Input,
    Draw
};


int main()
{
    auto fourier_resolution = 10UL;
    auto app                = App{{.dims{1800, 900}}};
    auto current_mode       = Mode::Input;
    auto fourier_state      = FourierState{};

    const auto& mouse = app.mouse;

    auto points = std::vector<complex>{};

    const auto refresh = [&]
    {
        current_mode = Mode::Draw;
        print("\nPoints: ", points.size());
        fourier_state.refresh(points, fourier_resolution);
        print();
    };

    app.keymap.push_back(Keyboard::OnKeyDown{{Keyboard::Key::Enter}, refresh});


    const auto draw_curve = [&]
    {
        if (points.size() < 2) { return; }

        for (auto i : range(points.size() - 1UL))
        {
            app.draw_line_segment({from_complex(points[i]), from_complex(points[i + 1UL])},
                                  "#c31432"_c, 0.15);
        }
    };

    auto mouse_click_previous = false;
    auto time                 = 0.0;

    const auto run = [&]
    {
        const auto mouse_pos = app.transform.apply_inverse(mouse.current_pos);
        if (mouse.left && !mouse_click_previous) { points.push_back(to_complex(mouse_pos)); }
        app.fill("#240b36"_c);
        draw_curve();

        if (current_mode == Mode::Draw) { fourier_state.draw(app, time); }
        mouse_click_previous = mouse.left;
        time += 1;
    };

    app.run(run);
}
