/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/core/concepts.hpp"

TEST_CASE("core.concepts", "vector2")
{
    static_assert(sm::concepts::Integral<sm::u8>);
    static_assert(sm::concepts::Integral<sm::u16>);
    static_assert(sm::concepts::Integral<sm::u32>);
    static_assert(sm::concepts::Integral<sm::u64>);

    static_assert(sm::concepts::Integral<sm::i8>);
    static_assert(sm::concepts::Integral<sm::i16>);
    static_assert(sm::concepts::Integral<sm::i32>);
    static_assert(sm::concepts::Integral<sm::i64>);


    static_assert(sm::concepts::Number<sm::u8>);
    static_assert(sm::concepts::Number<sm::u16>);
    static_assert(sm::concepts::Number<sm::u32>);
    static_assert(sm::concepts::Number<sm::u64>);

    static_assert(sm::concepts::Number<sm::i8>);
    static_assert(sm::concepts::Number<sm::i16>);
    static_assert(sm::concepts::Number<sm::i32>);
    static_assert(sm::concepts::Number<sm::i64>);

    static_assert(sm::concepts::Number<sm::f32>);
    static_assert(sm::concepts::Number<sm::f64>);
}
