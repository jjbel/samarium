/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/math/Vector2.hpp"
#include "samarium/math/vector_math.hpp"
#include "samarium/samarium.hpp"

#include "catch2/catch_test_macros.hpp"

using namespace sm;
using namespace sm::literals;

TEST_CASE("Vector2 Literals")
{
    const auto a_x = 1.0_x;
    const auto b_x = Vector2{1.0, 0};
    REQUIRE(a_x == b_x);

    const auto a_y = 1.0_y;
    const auto b_y = Vector2{0, 1.0};
    REQUIRE(a_y == b_y);
}

TEST_CASE("Vector2")
{
    static_assert(std::is_same_v<Vector2::value_type, f64>);

    SECTION("x vector")
    {
        const auto a = Vector2{1.0, 0.0};
        REQUIRE(math::almost_equal(a.length(), 1.0));
        REQUIRE(math::almost_equal(a.length_sq(), 1.0));
        REQUIRE(math::almost_equal(a.angle(), 0.0));
        REQUIRE(math::almost_equal(a.slope(), 0.0));
    }

    SECTION("xy vector")
    {
        const auto b = Vector2{1.0, 1.0};
        REQUIRE(math::almost_equal(b.length(), std::sqrt(2.0)));
        REQUIRE(math::almost_equal(b.length_sq(), 2.0));
        REQUIRE(math::almost_equal(b.angle(), math::to_radians(45.0)));
        REQUIRE(math::almost_equal(b.slope(), 1.0));
    }

    SECTION("y vector")
    {
        const auto c = Vector2{0.0, 1.0};
        REQUIRE(math::almost_equal(c.length(), 1.0));
        REQUIRE(math::almost_equal(c.length_sq(), 1.0));
        REQUIRE(math::almost_equal(c.angle(), math::to_radians(90.0)));
    }

    SECTION("origin vector")
    {
        const auto d = Vector2{0.0, 0.0};
        REQUIRE(math::almost_equal(d.length(), 0.0));
        REQUIRE(math::almost_equal(d.length_sq(), 0.0));
    }
};

TEST_CASE("geometry")
{
    SECTION("intersection")
    {
        SECTION("free")
        {
            const auto a = math::intersection(LineSegment{{-1.0, 0.0}, {1.0, 0.0}},
                                              LineSegment{{0.0, 1.0}, {0.0, -1.0}});
            REQUIRE(a.has_value());
            REQUIRE(*a == Vector2{});

            const auto b = math::intersection(LineSegment{{-1.0, -1.0}, {1.0, 1.0}},
                                              LineSegment{{1.0, -1.0}, {-1.0, 1.0}});
            REQUIRE(b.has_value());
            REQUIRE(*b == Vector2{});

            const auto c = math::intersection(LineSegment{{}, {0.0, 1.0}},
                                              LineSegment{{1.0, 0.0}, {1.0, 1.0}});
            REQUIRE(!c.has_value());
        }

        SECTION("clamped")
        {
            const auto a = math::clamped_intersection(LineSegment{{-1.0, 0.0}, {1.0, 0.0}},
                                                      LineSegment{{0.0, 1.0}, {0.0, -1.0}});
            REQUIRE(a.has_value());
            REQUIRE(*a == Vector2{});

            const auto b = math::clamped_intersection(LineSegment{{-1.0, -1.0}, {1.0, 1.0}},
                                                      LineSegment{{1.0, -1.0}, {-1.0, 1.0}});
            REQUIRE(b.has_value());
            REQUIRE(*b == Vector2{});

            const auto c = math::clamped_intersection(LineSegment{{-1.0, -1.0}, {-0.5, -0.5}},
                                                      LineSegment{{1.0, -1.0}, {0.5, -0.5}});
            REQUIRE(!c.has_value());

            const auto d = math::clamped_intersection(LineSegment{{-1.0, 0.0}, {-0.5, 0.0}},
                                                      {{0.0, 1.0}, {0.0, 0.5}});
            REQUIRE(!d.has_value());

            const auto e =
                math::clamped_intersection(LineSegment{{}, {0.0, 1.0}}, {{1.0, 0.0}, {1.0, 1.0}});
            REQUIRE(!e.has_value());
        }
    }

    SECTION("area")
    {
        SECTION("Circle")
        {
            REQUIRE(math::area(Circle{}) == 0.0);
            REQUIRE(math::almost_equal(math::area(Circle{.radius = 12.0}), 452.3893421169302));
        }

        SECTION("BoundingBox")
        {
            REQUIRE(math::area(BoundingBox<double>{}) == 0.0);
            REQUIRE(math::area(BoundingBox<double>{{-10.0, -11.0}, {12.0, 13.0}}) == 528.0);
        }
    }
}
