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
struct Sync
{
    GLsync handle{};

    void lock()
    {
        if (handle) { glDeleteSync(handle); }
        handle = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }

    void wait()
    {
        if (handle != nullptr)
        {
            while (true)
            {
                GLenum waitReturn = glClientWaitSync(handle, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
                if (waitReturn == GL_ALREADY_SIGNALED || waitReturn == GL_CONDITION_SATISFIED)
                {
                    return;
                }
            }
        }
    }
};
} // namespace sm::gl
