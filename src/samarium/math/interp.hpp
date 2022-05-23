/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../graphics/Color.hpp"
#include "../math/math.hpp"

#include "Extents.hpp"

namespace sm::interp
{

/**
 * @brief               Sigmoid-like smoothing
 * @param  value        Input value in the range [0, 1]
 * From https://easings.net/#easeOutElastic
 */
[[nodiscard]] constexpr auto ease_out_elastic(f64 value)
{
    if (value == 0.0) { return 0.0; }
    else if (value == 1.0)
    {
        return 1.0;
    }
    else
    {
        return std::pow(2.0, -10.0 * value) * std::sin((value * 10 - 0.75) * math::two_thirds_pi) +
               1;
    }
}

/**
 * @brief               Sigmoid-like smoothing
 * @param  value        Input value in the range [0, 1]
 */
[[nodiscard]] constexpr auto smooth_step(f64 value) { return value * value * (3.0 - 2.0 * value); }

/**
 * @brief               Smooth step but more!
 * @param  value        Input value in the range [0, 1]
 */
[[nodiscard]] constexpr auto smoother_step(f64 value)
{
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
}

// credit: https://gist.github.com/Bleuje/0917441d809d5eccf4ddcfc6a5b787d9

/**
 * @brief               Smooth a value with arbitrary smoothing
 *
 * @param  value        Input value in the range [0, 1]
 * @param  factor       Strength of smoothing: 1.0 is linear, higher values are smoother, values in
 * [0, 1) are inverse smoothing
 */
[[nodiscard]] inline auto smooth(f64 value, f64 factor = 2.0)
{
    if (value < 0.5) { return 0.5 * std::pow(2.0 * value, factor); }
    else
    {
        return 1.0 - 0.5 * std::pow(2.0 * (1.0 - value), factor);
    }
}

/**
 * @brief               Check if a value is within asome Extents
 *
 * @param  value        Input value
 * @param  range_       Input range
 */
template <typename T> [[nodiscard]] constexpr auto in_range(T value, Extents<T> range_)
{
    return range_.contains(value);
}

/**
 * @brief               Ensure a value is within some Extents
 *
 * @param  value
 * @param  range_
 */
template <typename T> [[nodiscard]] constexpr auto clamp(T value, Extents<T> range_) noexcept
{
    return range_.clamp(value);
}

/**
 * @brief               Linearly interpolate a factor in a range
 * @param  factor
 * @param  range_
 */
template <typename T> [[nodiscard]] constexpr auto lerp(f64 factor, Extents<T> range_)
{
    return range_.lerp(factor);
}

/**
 * @brief               Lerp, but clamp the factor in [0, 1]
 * @param  factor
 * @param  range_
 */
template <typename T> [[nodiscard]] constexpr auto clamped_lerp(f64 factor, Extents<T> range_)
{
    return range_.clamped_lerp(factor);
}

/**
 * @brief               Lerp between the RGBA values of 2 Colors
 * @param  factor
 * @param  from
 * @param  to
 */
[[nodiscard]] constexpr auto lerp_rgb(f64 factor, Color from, Color to)
{
    return Color{static_cast<u8>(
                     lerp(factor, Extents<f64>{static_cast<f64>(from.r), static_cast<f64>(to.r)})),
                 static_cast<u8>(
                     lerp(factor, Extents<f64>{static_cast<f64>(from.g), static_cast<f64>(to.g)})),
                 static_cast<u8>(
                     lerp(factor, Extents<f64>{static_cast<f64>(from.b), static_cast<f64>(to.b)})),
                 static_cast<u8>(
                     lerp(factor, Extents<f64>{static_cast<f64>(from.a), static_cast<f64>(to.a)}))};
}

/**
 * @brief               Find the factor which lerps the value in the range
 * @param  value
 * @param  range_
 */
template <typename T> [[nodiscard]] constexpr auto lerp_inverse(f64 value, Extents<T> range_)
{
    return range_.lerp_inverse(value);
}

// https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

/**
 * @brief               Map a value from an input range to an output range
 * @tparam T            Type of value
 * @tparam Output       Cast the result to Output
 * @param  value
 * @param  from
 * @param  to
 */
template <typename T, typename Output = T>
[[nodiscard]] constexpr auto map_range(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output>(to.min + (value - from.min) * to.size() / from.size());
}

/**
 * @brief               Map range, but clamp the value to the output range
 * @tparam T            Type of value
 * @tparam Output       Cast the result to Output
 * @param  value
 * @param  from
 * @param  to
 */
template <typename T, typename Output = T>
[[nodiscard]] constexpr auto map_range_clamp(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output>(to.min + (from.clamp(value) - from.min) * to.size() / from.size());
}

/**
 * @brief               Make a lambda which maps its argument from and to fixed ranges
 * @tparam T            Type of value
 * @tparam Output       Cast the result to Output
 * @param  from
 * @param  to
 */
template <typename T, typename Output = T>
[[nodiscard]] constexpr auto make_mapper(Extents<T> from, Extents<T> to)
{
    return [from_min = from.min, from_range = from.size(), to_range = to.size(),
            to_min = to.min](T value)
    { return static_cast<Output>(to_min + (value - from_min) * to_range / from_range); };
}

/**
 * @brief               Make a lambda which maps its argument from and to fixed ranges, with
 * clamping
 * @tparam T            Type of value
 * @tparam Output       Cast the result to Output
 * @param  from
 * @param  to
 */
template <typename T, typename Output = T>
[[nodiscard]] constexpr auto make_clamped_mapper(Extents<T> from, Extents<T> to)
{
    return [from, from_min = from.min, from_max = from.max, from_range = from.max - from.min,
            to_range = to.size(), to_min = to.min](T value) {
        return static_cast<Output>(to_min + (from.clamp(value) - from_min) * to_range / from_range);
    };
}

} // namespace sm::interp

namespace sm::concepts
{
/**
 * @brief               Can T be called with an f64 (commonly in the range [0, 1] to use in
 * interpolation)
 */
template <typename T>
concept Interpolator = std::invocable<T, f64>;
} // namespace sm::concepts
