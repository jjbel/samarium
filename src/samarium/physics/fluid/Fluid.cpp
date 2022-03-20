/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "Fluid.hpp"

namespace sm
{
void Fluid::lin_solve(ScalarField& x_now, ScalarField& x_prev, f64 a, f64 weight, u64 iter)
{
    for (auto k : math::range({}, iter))
    {
        std::ignore = k;

        for (auto j : math::range(1, size - 1))
        {
            for (auto i : math::range(1, size - 1))
            {
                const auto value = x_prev[{i, j}] + a * (x_now[{i + 1, j}] + x_now[{i - 1, j}] +
                                                         x_now[{i, j + 1}] + x_now[{i, j - 1}]);

                x_now[{i, j}] = value / weight;
            }
        }

        set_bnd();
    }
}

/* void Fluid::lin_solve(Dual<VectorField> x, f64 a, f64 weight, u64 iter)
{
    for (auto k : math::range({}, iter))
    {
        for (auto j : math::range(1, size - 1))
        {
            for (auto i : math::range(1, size - 1))
            {
                const auto value = x.prev[{i, j}] + a * (x.now[{i + 1, j}] + x.now[{i - 1, j}] +
                                                         x.now[{i, j + 1}] + x.now[{i, j - 1}]);

                x.now[{i, j}] = value / weight;
            }
        }

        set_bnd();
    }
} */

void Fluid::add_density(Indices pos, f64 amount) { this->density.now[pos] += amount; }

void Fluid::add_velocity(Indices pos, Vector2 amount) { this->vel.now[pos] += amount; }

void Fluid::update()
{
    // diffuse(Vx0, Vx, visc, dt);
    // diffuse(Vy0, Vy, visc, dt);

    // project(Vx0, Vy0, Vx, Vy);

    // advect(Vx, Vx0, Vx0, Vy0, dt);
    // advect(Vy, Vy0, Vx0, Vy0, dt);

    // project(Vx, Vy, Vx0, Vy0);

    // diffuse(s, density, diff, dt);
    // advect(density, s, Vx, Vy, dt);
}

void Fluid::diffuse()
{
    const auto factor = dt * diff * static_cast<f64>((size - 2UL) * (size - 2UL));
    lin_solve(density.now, density.prev, factor, 1 + 4 * factor);
}

void Fluid::project(VectorField& veloc, ScalarField p, ScalarField div)
{
    for (auto j : math::range(1, size - 1))
    {
        for (auto i : math::range(1, size - 1))
        {
            const auto value = -0.5 / static_cast<f64>(size) *
                               (veloc[{i + 1, j}].x - veloc[{i - 1, j}].x + veloc[{i, j + 1}].y -
                                veloc[{i, j - 1}].y);

            div[{i, j}] = value;
            p[{i, j}]   = 0;
        }
    }

    set_bnd();

    lin_solve(p, div, 1, 4);

    for (auto j : math::range(1, size - 1))
    {
        for (auto i : math::range(1, size - 1))
        {
            veloc[{i, j}].x -= 0.5 * (p[{i + 1, j}] - p[{i - 1, j}]) * size;
            veloc[{i, j}].y -= 0.5 * (p[{i, j + 1}] - p[{i, j - 1}]) * size;
        }
    }

    set_bnd();
}

void Fluid::advect(Dual<ScalarField>& d, VectorField& veloc)
{
    f64 dtx = dt * static_cast<f64>(size - 2UL);
    f64 dty = dt * static_cast<f64>(size - 2UL);

    f64 s0, s1, t0, t1;

    f64 Nf64 = size;

    for (auto j : math::range(1, size - 1))
    {
        for (auto i : math::range(1, size - 1))
        {
            f64 tmp1 = dtx * veloc[{i, j}].x;
            f64 tmp2 = dty * veloc[{i, j}].y;
            f64 x    = static_cast<f64>(i) - tmp1;
            f64 y    = static_cast<f64>(j) - tmp2;

            if (x < 0.5) x = 0.5;
            if (x > Nf64 + 0.5) x = Nf64 + 0.5;

            f64 i0 = std::floor(x);
            f64 i1 = i0 + 1.0;

            if (y < 0.5) y = 0.5;
            if (y > Nf64 + 0.5) y = Nf64 + 0.5;
            f64 j0 = std::floor(y);

            f64 j1 = j0 + 1.0;

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            // TODO DOUBLE CHECK THIS!!!
            const auto val = s0 * (t0 * d.prev[{static_cast<u64>(i0), static_cast<u64>(j0)}] +
                                   t1 * d.prev[{static_cast<u64>(i0), static_cast<u64>(j1)}]) +
                             s1 * (t0 * d.prev[{static_cast<u64>(i1), static_cast<u64>(j0)}] +
                                   t1 * d.prev[{static_cast<u64>(i1), static_cast<u64>(j1)}]);

            d.now[{i, j}] = val;
        }
    }

    set_bnd();
}
} // namespace sm
