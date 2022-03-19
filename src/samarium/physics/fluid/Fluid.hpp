/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../../graphics/Image.hpp"
#include "../../math/Dual.hpp"

// From: https://youtu.be/alhpH6ECFvQ


namespace sm
{
struct Fluid
{
    Dual<RealField> density;
    Dual<VectorField> vel;

    const size_t size;

    const f64 dt{0.1};
    const f64 diff{1};
    const f64 visc{1};


    Fluid(size_t size_ = 128ul)
        : density{.now = RealField(Dimensions{size_, size_})},
          vel{.now = VectorField(Dimensions{size_, size_})}, size{size_}
    {
    }

    void add_density(Indices pos, f64 amount);

    void add_velocity(Indices pos, Vector2 amount);

    void update();

    void diffuse();

    void project(VectorField& veloc, RealField p, RealField div);

    void advect(Dual<RealField>& d, VectorField& veloc, f64 dt);

    void lin_solve(RealField& x_now, RealField& x_prev, f64 a, f64 weight, u64 iter = 1UL);

    void set_bnd();
};
} // namespace sm
