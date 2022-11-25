/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept> // for runtime_error

#include "fmt/core.h" // for format
#include "glad/glad.h"

#include "samarium/core/types.hpp"     // for u32
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/util/Result.hpp"    // for Result

#include "Texture.hpp" // for Texture

namespace sm::gl
{
struct Framebuffer
{
    u32 handle{};

    explicit Framebuffer(const Texture& texture);

    Framebuffer(const Framebuffer&) = delete;

    auto operator=(const Framebuffer&) -> Framebuffer& = delete;

    Framebuffer(Framebuffer&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(Framebuffer&& other) noexcept -> Framebuffer&
    {
        if (this != &other)
        {
            glDeleteVertexArrays(1, &handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    void bind() const;

    void bind_texture(const Texture& texture) const;

    void unbind() const;

    void clear(Color color) const;

    ~Framebuffer();
};
} // namespace sm::gl


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_FRAMEBUFFER_IMPL)

#include "glad/glad.h"

#include "samarium/core/inline.hpp"

#include "Framebuffer.hpp"

namespace sm::gl
{
SM_INLINE Framebuffer::Framebuffer(const Texture& texture)
{
    glCreateFramebuffers(1, &handle);
    bind_texture(texture);
}

SM_INLINE void Framebuffer::bind() const { glBindFramebuffer(GL_FRAMEBUFFER, handle); }

SM_INLINE void Framebuffer::unbind() const { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

SM_INLINE void Framebuffer::bind_texture(const Texture& texture) const
{

    glNamedFramebufferTexture(handle, GL_COLOR_ATTACHMENT0, texture.handle, 0);

    const auto status = glCheckNamedFramebufferStatus(handle, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::runtime_error{fmt::format("Framebuffer intialization error: {}", status)};
    }
}

SM_INLINE void Framebuffer::clear(Color color) const
{
    glClearNamedFramebufferfv(handle, GL_COLOR, 0, color.to_float_array().data());
}

SM_INLINE Framebuffer::~Framebuffer() { glDeleteFramebuffers(1, &handle); }
} // namespace sm::gl

#endif
