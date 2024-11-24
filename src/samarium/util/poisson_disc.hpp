/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <vector> // for vector

#include "samarium/core/types.hpp" // for f64, u64, i64, i32
#include "samarium/math/Box2.hpp"  // for Box2
// #include "samarium/math/Extents.hpp"     // for Extents
#include "samarium/math/Vec2.hpp"            // for Vec2
#include "samarium/math/math.hpp"            // for pi
#include "samarium/util/RandomGenerator.hpp" // for RandomGenerator

namespace sm::poisson_disc
{
// distribute points such that they are at least radius away from each other
// reference: Sebastian Lague: https://youtu.be/7WcmyxyFO7o
//
// To, place discs of radius r in a region such that they don't overlap
// Their centres are 2r away, so run with 2 * radius
[[nodiscard]] auto uniform(RandomGenerator& rng,
                           f64 radius,
                           Box2<f64> sample_region,
                           u64 sample_count) -> std::vector<Vec2>;
} // namespace sm::poisson_disc


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_POISSON_DISC_IMPL)

#include "samarium/math/Box2.hpp"        // for Box2
#include "samarium/math/loop.hpp"        // for start_end
#include "samarium/math/vector_math.hpp" // for within_distance
#include "samarium/util/Grid.hpp"        // for Grid

namespace sm::poisson_disc
{
[[nodiscard]] auto uniform_is_valid(Vec2 candidate,
                                    Vec2 sample_region,
                                    f64 cell_size,
                                    f64 radius,
                                    const std::vector<Vec2>& points,
                                    const Grid<i32>& grid)
{
    // TODO too many casts bw signed and unsigned

    // TODO use Box2
    // TODO should check > radius, not > 0 ?
    if (!(candidate.x >= 0.0 && candidate.x < sample_region.x && candidate.y >= 0.0 &&
          candidate.y < sample_region.y))
    {
        return false;
    }

    const auto cell_x         = static_cast<i64>(candidate.x / cell_size);
    const auto cell_y         = static_cast<i64>(candidate.y / cell_size);
    const auto search_start_x = static_cast<u64>(math::max<i64>(0L, cell_x - 2L));
    const auto search_end_x =
        static_cast<u64>(math::min<i64>(cell_x + 2L, static_cast<i64>(grid.dims.x)));
    const auto search_start_y = static_cast<u64>(math::max<i64>(0L, cell_y - 2));
    const auto search_end_y =
        static_cast<u64>(math::min<i64>(cell_y + 2L, static_cast<i64>(grid.dims.y)));

    for (auto x : loop::start_end(search_start_x, search_end_x))
    {
        for (auto y : loop::start_end(search_start_y, search_end_y))
        {
            // grid stores 1-based index
            const auto point_index = grid[Indices{x, y}] - 1;
            // TODO using 2*radius or radius. points vs discs
            if (point_index != -1 &&
                math::within_distance(candidate, points[static_cast<u64>(point_index)], radius))
            {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] auto uniform(RandomGenerator& rng,
                           f64 radius,
                           Box2<f64> sample_region,
                           u64 sample_count) -> std::vector<Vec2>
{
    // at most one disc can fit in one cell
    const auto cell_size          = radius / std::numbers::sqrt2;
    const auto sample_region_size = sample_region.diagonal();

    // grid[x, y] is the index of that cell's particle, in the points vector
    // we only ever push_back to points, so indices are stable
    // use ceil so to ensure the grid covers the entire region
    auto grid = Grid<i32>{{static_cast<u64>(std::ceil(sample_region_size.x / cell_size)),
                           static_cast<u64>(std::ceil(sample_region_size.y / cell_size))}};

    // when we add a point, try to spawn more points around it
    // if it gets too crowded, remove it from spawn_points
    auto points       = std::vector<Vec2>();
    auto spawn_points = std::vector<Vec2>();

    // start spawning from the middle of the region
    spawn_points.push_back(sample_region_size / 2.0);
    // TODO reserve before push_back

    while (!spawn_points.empty())
    {
        // randomly choose a spawn point
        const auto spawn_index  = rng.range<u64>({0UL, spawn_points.size() - 1UL});
        const auto spawn_centre = spawn_points[spawn_index];

        // was at least 1 point added from this spawn point
        auto candidate_accepted = false;

        for (auto i : loop::end(sample_count))
        {
            std::ignore = i;

            // min is radius so that it doesn't overlap with spawn_centre's disc
            // TODO shouldn't min be 2*radius
            const auto candidate = spawn_centre + rng.polar_vector({radius, 2 * radius});

            if (!uniform_is_valid(candidate, sample_region_size, cell_size, radius, points, grid))
            {
                continue;
            }
            points.push_back(candidate);
            spawn_points.push_back(candidate);

            // 1-based indexing, 0 means cell is empty
            grid[{static_cast<u64>(candidate.x / cell_size),
                  static_cast<u64>(candidate.y / cell_size)}] = static_cast<i32>(points.size());

            candidate_accepted = true;
            break;
        }

        if (!candidate_accepted)
        {
            spawn_points.erase(spawn_points.begin() + static_cast<i64>(spawn_index));
        }
    }

    for (auto& point : points) { point += sample_region.min; }
    return points;
}
} // namespace sm::poisson_disc
#endif
