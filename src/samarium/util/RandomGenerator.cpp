/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "RandomGenerator.hpp"
#include "../math/geometry.hpp"
#include "../math/interp.hpp"

namespace sm
{
namespace detail
{
[[nodiscard]] auto poisson_is_valid(Vector2 candidate,
                                    Vector2 sampleRegionSize,
                                    f64 cellSize,
                                    f64 radius,
                                    const std::vector<Vector2>& points,
                                    const Grid<i32>& grid)
{
    if (!(candidate.x >= 0.0 && candidate.x < sampleRegionSize.x && candidate.y >= 0.0 &&
          candidate.y < sampleRegionSize.y))
    {
        return false;
    }

    const auto cellX        = static_cast<i64>(candidate.x / cellSize);
    const auto cellY        = static_cast<i64>(candidate.y / cellSize);
    const auto searchStartX = static_cast<u64>(std::max(0L, cellX - 2));
    const auto searchEndX   = static_cast<u64>(std::min(cellX + 2, static_cast<i64>(grid.dims.x)));
    const auto searchStartY = static_cast<u64>(std::max(0L, cellY - 2));
    const auto searchEndY   = static_cast<u64>(std::min(cellY + 2, static_cast<i64>(grid.dims.y)));

    for (auto x : range(searchStartX, searchEndX))
    {
        for (auto y : range(searchStartY, searchEndY))
        {
            const auto pointIndex = grid[Indices{x, y}] - 1;
            if (pointIndex != -1 &&
                math::within_distance(candidate, points[static_cast<u64>(pointIndex)], radius))
            {
                return false;
            }
        }
    }
    return true;
}
} // namespace detail

void RandomGenerator::resize(u64 new_size)
{
    if (new_size < cache.size()) { return; }
    cache.resize(new_size);
    std::generate(cache.begin(), cache.end(), [this] { return this->next_scaled(); });
}

void RandomGenerator::reseed(u64 new_seed)
{
    inc           = new_seed;
    current_index = 0UL;
    std::generate(cache.begin(), cache.end(), [this] { return this->next_scaled(); });
}

[[nodiscard]] auto RandomGenerator::next() noexcept -> u64
{
    const auto oldstate = this->state;
    // Advance internal state
    this->state = oldstate * magic_number + (this->inc | 1);
    // Calculate output function (XSH RR), uses old state for max ILP
    const auto xorshifted = static_cast<u32>(((oldstate >> 18UL) ^ oldstate) >> 27UL);
    const auto rot        = static_cast<u32>(oldstate >> 59UL);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

[[nodiscard]] auto RandomGenerator::next_scaled() noexcept -> f64
{
    return static_cast<f64>(this->next()) / static_cast<f64>(std::numeric_limits<u32>::max());
}

[[nodiscard]] auto RandomGenerator::random() -> f64
{
    if (current_index < cache.size()) return cache[current_index++];

    return this->next_scaled();
}

[[nodiscard]] auto RandomGenerator::vector(const BoundingBox<f64>& bounding_box) noexcept -> Vector2
{
    return Vector2{this->range<f64>({bounding_box.min.x, bounding_box.max.x}),
                   this->range<f64>({bounding_box.min.y, bounding_box.max.y})};
}

[[nodiscard]] auto RandomGenerator::polar_vector(Extents<f64> radius_range,
                                                 Extents<f64> angle_range) noexcept -> Vector2
{
    return Vector2::from_polar({.length = this->range<f64>({radius_range.min, radius_range.max}),
                                .angle  = this->range<f64>({angle_range.min, angle_range.max})});
}

[[nodiscard]] auto RandomGenerator::poisson_disc_points(f64 radius,
                                                        Vector2 sample_region_size,
                                                        u64 sample_count) -> std::vector<Vector2>
{
    const f64 cellSize = radius / std::numbers::sqrt2;

    auto grid = Grid<i32>{{static_cast<u64>(std::ceil(sample_region_size.x / cellSize)),
                           static_cast<u64>(std::ceil(sample_region_size.y / cellSize))}};

    auto points      = std::vector<Vector2>();
    auto spawnPoints = std::vector<Vector2>();

    spawnPoints.push_back(sample_region_size / 2.0);

    while (!spawnPoints.empty())
    {
        const auto spawnIndex  = this->range<u64>({0UL, spawnPoints.size() - 1});
        auto spawnCentre       = spawnPoints[spawnIndex];
        auto candidateAccepted = false;

        for (auto i : sm::range(sample_count))
        {
            std::ignore          = i;
            const auto candidate = spawnCentre + this->polar_vector({radius, 2 * radius});

            if (detail::poisson_is_valid(candidate, sample_region_size, cellSize, radius, points,
                                         grid))
            {
                points.push_back(candidate);
                spawnPoints.push_back(candidate);
                grid[{static_cast<u64>(candidate.x / cellSize),
                      static_cast<u64>(candidate.y / cellSize)}] = static_cast<i32>(points.size());

                candidateAccepted = true;
                break;
            }
        }

        if (!candidateAccepted)
        {
            spawnPoints.erase(spawnPoints.begin() + static_cast<i64>(spawnIndex));
        }
    }

    return points;
}
} // namespace sm
