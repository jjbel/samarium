/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include <array>
#include <span>

#include "samarium/core/types.hpp"   // for usize, f64
#include "samarium/math/Vector2.hpp" // for Vector2_t
#include "samarium/math/loop.hpp"    // for start_end

namespace sm
{
template <usize degree, typename T = f64> struct Polynomial
{
    static constexpr auto Degree = degree;
    using value_type             = T;

    std::array<T, degree + 1> coeffs{};

    [[nodiscard]] constexpr auto operator()(T value)
    {
        auto sum           = coeffs[0];
        auto current_value = value;

        for (auto i : loop::start_end(u64{1}, degree + 1))
        {
            sum += coeffs[i] * current_value;
            current_value *= value;
        }

        return sum;
    }
};

template <typename... Ts> [[nodiscard]] constexpr auto make_polynomial(Ts... coeffs)
{
    using type = std::tuple_element_t<0, std::tuple<Ts...>>;
    return Polynomial<sizeof...(coeffs) - 1, type>{{coeffs...}};
}

template <usize Degree, typename T = f64>
[[nodiscard]] constexpr auto polynomial_from_roots(std::span<T, Degree> roots,
                                                   T coeff = T{1.0}) -> Polynomial<Degree, T>
{
    if constexpr (Degree == 1) { return {{-coeff * roots[0], coeff}}; }
    if constexpr (Degree == 2)
    {
        return {{coeff * roots[0] * roots[1], -coeff * (roots[0] + roots[1]), coeff}};
    }
    // TODO
}

template <usize Degree, typename T = f64>
[[nodiscard]] constexpr auto polynomial_from_points(const std::array<T, Degree>& points)
{
    // TODO
}
} // namespace sm
