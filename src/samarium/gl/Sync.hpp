/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "glad/glad.h"

#include "gl.hpp"

namespace sm::gl
{
// https://www.cppstories.com/2015/01/persistent-mapped-buffers-benchmark/
struct Sync
{
    GLsync handle{};

    void fence_sync()
    {
        if (handle != nullptr) { glDeleteSync(handle); }
        handle = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }

    auto wait()
    {
        if (handle == nullptr) { return -1; }
        auto count = i32{-1};
        while (true)
        {
            count++;
            GLenum waitReturn = glClientWaitSync(handle, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (waitReturn == GL_ALREADY_SIGNALED || waitReturn == GL_CONDITION_SATISFIED)
            {
                return count;
            }
        }
    }
};
} // namespace sm::gl
