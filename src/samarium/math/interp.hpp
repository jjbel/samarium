/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../core/types.hpp"
#include "../graphics/Color.hpp"

#include "./Extents.hpp"

namespace sm::interp
{
[[nodiscard]] constexpr auto smooth_step()
{
    return [](auto x) { return x * x * (3. - 2. * x); };
}

[[nodiscard]] constexpr auto smoother_step()
{
    return [](auto x) { return x * x * x * (x * (x * 6. - 15.) + 10.); };
}

template <typename T>
[[nodiscard]] constexpr auto in_range(T value, Extents<T> range)
{
    return range.contains(value);
}

template <typename T>
[[nodiscard]] constexpr auto clamp(T value, Extents<T> range) noexcept
{
    return range.clamp(value);
}

template <typename T>
[[nodiscard]] constexpr auto lerp(f64 factor, Extents<T> range)
{
    return range.lerp(factor);
}

template <typename T>
[[nodiscard]] constexpr auto clamped_lerp(f64 factor, Extents<T> range)
{
    return range.clamped_lerp(factor);
}

[[nodiscard]] constexpr auto lerp_rgb(f64 factor, Color from, Color to)
{
    return Color{
        static_cast<u8>(lerp(factor, Extents<f64>{static_cast<f64>(from.r),
                                                  static_cast<f64>(to.r)})),
        static_cast<u8>(lerp(factor, Extents<f64>{static_cast<f64>(from.g),
                                                  static_cast<f64>(to.g)})),
        static_cast<u8>(lerp(factor, Extents<f64>{static_cast<f64>(from.b),
                                                  static_cast<f64>(to.b)})),
        static_cast<u8>(lerp(factor, Extents<f64>{static_cast<f64>(from.a),
                                                  static_cast<f64>(to.a)}))};
}

template <typename T>
[[nodiscard]] constexpr auto lerp_inverse(f64 value, Extents<T> range)
{
    return range.lerp_inverse(value);
}

// https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto map_range(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output_t>(to.min +
                                 (value - from.min) * to.size() / from.size());
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto
map_range_clamp(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output_t>(to.min + (from.clamp(value) - from.min) *
                                              to.size() / from.size());
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto make_mapper(Extents<T> from, Extents<T> to)
{
    return [from_min = from.min, from_max = from.max, from_range = from.size(),
            to_range = to.size(), to_min = to.min](T value)
    {
        return static_cast<Output_t>(to_min +
                                     (value - from_min) * to_range / from_range);
    };
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto make_clamped_mapper(Extents<T> from, Extents<T> to)
{
    return [from, from_min = from.min, from_max = from.max,
            from_range = from.max - from.min, to_range = to.size(),
            to_min = to.min](T value)
    {
        return static_cast<Output_t>(to_min + (from.clamp(value) - from_min) *
                                                  to_range / from_range);
    };
}

} // namespace sm::interp

namespace sm::concepts
{
template <typename T>
concept Interpolator = std::invocable<T, f64>;
}
