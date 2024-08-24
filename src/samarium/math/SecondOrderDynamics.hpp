/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp" // for f64
#include "samarium/math/math.hpp"  // for constants

namespace sm
{
/**
 * @brief               Make a value follow an input value using a second order
 * differential equation
 *
 */
template <typename T> struct SecondOrderDynamics
{
    T value;
    T previous_input;
    T vel{};
    f64 k1;
    f64 k2;
    f64 k3;

    /**
     * @brief               Start responding to `initial_input_value`
     *
     * @param  initial_input_value
     * @param  frequency    Frequency of oscillation. Range: [0, inf)
     * @param  damping      [0, 1.0): with ocscillation. [1.0, inf): asymptotic approach
     * @param  response    at 0, takes time to respond. At 1.0, immediate response. When > 1:
     * overshoots target. < 1 anticipates motion
     */
    explicit SecondOrderDynamics(T initial_input_value = {},
                                 f64 frequency         = 4.0,
                                 f64 damping           = 0.3,
                                 f64 response          = 0.0)
        : value{initial_input_value}, previous_input{initial_input_value}
    {
        update_parameters(frequency, damping, response);
    }

    /**
     * @brief               Update the equation parameters
     *
     * @param  frequency    Frequency of oscillation. Range: [0, inf)
     * @param  damping      [0, 1.0): with ocscillation. [1.0, inf): asymptotic approach
     * @param  response    at 0, takes time to respond. At 1.0, immediate response. When > 1:
     * overshoots target. < 1 anticipates motion
     */
    void update_parameters(f64 frequency, f64 damping, f64 response)
    {
        k1 = damping / (math::pi * frequency);
        k2 = 1.0 / math::power<2>(math::two_pi * frequency);
        k3 = response * damping / (math::two_pi * frequency);
    }

    /**
     * @brief               Update value and velocity by integrating
     *
     * @param  dt           Time step
     * @param  input_value  New input to track
     */

    void update(f64 dt, T input_value)
    {
        update(dt, input_value, (input_value - previous_input) / dt);
        previous_input = input_value;
    }

    /**
     * @brief               Update value and velocity by integrating
     *
     * @param  dt           Time step
     * @param  input_value  New input to track
     * @param  input_vel    New velocity
     */
    void update(f64 dt, T input_value, T input_vel)
    {
        value += vel * dt;
        vel += dt * (input_value + k3 * input_vel - value - k1 * vel) / k2;
    }
};
} // namespace sm
