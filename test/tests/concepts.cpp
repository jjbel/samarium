/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../ut.hpp"

#include "samarium/core/concepts.hpp"

boost::ut::suite concepts = []
{
    using namespace boost::ut;
    "core.concepts"_test = []
    {
        expect(sm::concepts::Integral<sm::u8>);
        expect(sm::concepts::Integral<sm::u16>);
        expect(sm::concepts::Integral<sm::u32>);
        expect(sm::concepts::Integral<sm::u64>);

        expect(sm::concepts::Integral<sm::i8>);
        expect(sm::concepts::Integral<sm::i16>);
        expect(sm::concepts::Integral<sm::i32>);
        expect(sm::concepts::Integral<sm::i64>);

        expect(sm::concepts::Number<sm::u8>);
        expect(sm::concepts::Number<sm::u16>);
        expect(sm::concepts::Number<sm::u32>);
        expect(sm::concepts::Number<sm::u64>);

        expect(sm::concepts::Number<sm::i8>);
        expect(sm::concepts::Number<sm::i16>);
        expect(sm::concepts::Number<sm::i32>);
        expect(sm::concepts::Number<sm::i64>);

        expect(sm::concepts::Number<sm::f32>);
        expect(sm::concepts::Number<sm::f64>);
    };
};
