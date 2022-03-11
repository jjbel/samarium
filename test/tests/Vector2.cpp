/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../ut.hpp"

#include "../../src/samarium/math/Vector2.hpp"
#include "../../src/samarium/math/geometry.hpp"

using namespace boost::ut;

boost::ut::suite _Vector2 = []
{
    "util.format.Vector2"_test = []
    {
        using namespace std::literals;
        expect(fmt::format("{}", sm::Vector2{2, 3}) == "Vec( 2.000,  3.000)"s);
        expect(fmt::format("{}", sm::Indices{2, 3}) == "Vec(  2,   3)"s);
    };

    "math.Vector2.literals"_test = []
    {
        using namespace sm::literals;

        const auto a_x = 1.0_x;
        const auto b_x = sm::Vector2{1.0, 0};
        expect(a_x == b_x);

        const auto a_y = 1.0_y;
        const auto b_y = sm::Vector2{0, 1.0};
        expect(a_y == b_y);
    };

    "math.Vector2"_test = []
    {
        static_assert(std::is_same_v<sm::Vector2::value_type, sm::f64>);

        should("x vector") = [=]() mutable
        {
            const auto a = sm::Vector2{1.0, 0.0};
            expect(sm::math::almost_equal(a.length(), 1.0));
            expect(sm::math::almost_equal(a.length_sq(), 1.0));
            expect(sm::math::almost_equal(a.angle(), 0.0));
            expect(sm::math::almost_equal(a.slope(), 0.0));
        };

        should("xy vector") = [=]() mutable
        {
            const auto b = sm::Vector2{1.0, 1.0};
            expect(sm::math::almost_equal(b.length(), std::sqrt(2.0)));
            expect(sm::math::almost_equal(b.length_sq(), 2.0));
            expect(sm::math::almost_equal(b.angle(), sm::math::to_radians(45)));
            expect(sm::math::almost_equal(b.slope(), 1.0));
        };

        should("y vector") = [=]() mutable
        {
            const auto c = sm::Vector2{0.0, 1.0};
            expect(sm::math::almost_equal(c.length(), 1.0));
            expect(sm::math::almost_equal(c.length_sq(), 1.0));
            expect(sm::math::almost_equal(c.angle(), sm::math::to_radians(90)));
        };

        should("origin vector") = [=]() mutable
        {
            const auto d = sm::Vector2{0.0, 0.0};
            expect(sm::math::almost_equal(d.length(), 0.0));
            expect(sm::math::almost_equal(d.length_sq(), 0.0));
        };
    };

    "math.Vector2.geometry"_test = []
    {
        should("intersection") = [=]() mutable
        {
            should("free") = [=]() mutable
            {
                const auto a = sm::math::intersection({{-1.0, 0.0}, {1.0, 0.0}},
                                                      {{0.0, 1.0}, {0.0, -1.0}});
                expect(a.has_value());
                expect(*a == sm::Vector2{});

                const auto b = sm::math::intersection({{-1.0, -1.0}, {1.0, 1.0}},
                                                      {{1.0, -1.0}, {-1.0, 1.0}});
                expect(b.has_value());
                expect(*b == sm::Vector2{});

                const auto c = sm::math::intersection({{}, {0.0, 1.0}},
                                                      {{1.0, 0.0}, {1.0, 1.0}});
                expect(!c.has_value());
            };

            should("clamped") = [=]() mutable
            {
                const auto a = sm::math::clamped_intersection(
                    {{-1.0, 0.0}, {1.0, 0.0}}, {{0.0, 1.0}, {0.0, -1.0}});
                expect(a.has_value());
                expect(*a == sm::Vector2{});

                const auto b = sm::math::clamped_intersection(
                    {{-1.0, -1.0}, {1.0, 1.0}}, {{1.0, -1.0}, {-1.0, 1.0}});
                expect(b.has_value());
                expect(*b == sm::Vector2{});

                const auto c = sm::math::clamped_intersection(
                    {{-1.0, -1.0}, {-0.5, -0.5}}, {{1.0, -1.0}, {0.5, -0.5}});
                expect(!c.has_value());

                const auto d = sm::math::clamped_intersection(
                    {{-1.0, 0.0}, {-0.5, 0.0}}, {{0.0, 1.0}, {0.0, 0.5}});
                expect(!d.has_value());

                const auto e = sm::math::clamped_intersection(
                    {{}, {0.0, 1.0}}, {{1.0, 0.0}, {1.0, 1.0}});
                expect(!e.has_value());
            };
        };
    };
};
