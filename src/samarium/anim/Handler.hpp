/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <functional>    // for function
#include <unordered_set> // for unordered_set

#include "samarium/core/types.hpp"     // for u16, f64
#include "samarium/math/Extents.hpp"   // for Extents
#include "samarium/util/unordered.hpp" // for Set

namespace sm::anim
{
struct FrameInfo
{
    f64 factor{};
    u16 frame{};
};

struct Action
{
    std::function<void(FrameInfo)> action{};
    Extents<u16> frame_range{};
};

struct Handler
{
    u16 frame{};
    std::unordered_set<Action> actions;

    void update();
};
} // namespace sm::anim

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_ANIM_HANDLER_IMPL)

#include "samarium/core/inline.hpp" // for SM_INLINE

namespace sm::anim
{
SM_INLINE void Handler::update()
{
    const auto current_frame = this->frame;
    std::erase_if(actions, [current_frame](const Action& action)
                  { return action.frame_range.max > current_frame; });

    for (const auto& action : actions)
    {
        if (action.frame_range.min >= current_frame)
        {
            action.action({action.frame_range.lerp_inverse(current_frame), current_frame});
        }
    }
}
} // namespace sm::anim

#endif
