/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept>

#include "fmt/format.h"
#include "glad/glad.h"
#include "tl/expected.hpp"

#include "samarium/core/types.hpp"     // for u32
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/util/Expected.hpp"  // for Expected

namespace sm::gl
{
struct Framebuffer
{
    u32 handle{};

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

    [[nodiscard]] static auto make() -> Expected<Framebuffer, std::string>
    {
        auto handle = u32{};
        glCreateFramebuffers(1, &handle);

        auto framebuffer = Framebuffer{handle};
        return {std::move(framebuffer)};
    }

    [[nodiscard]] auto status() const -> u32;

    void bind() const;

    void clear(Color color) const;

    ~Framebuffer();

  private:
    Framebuffer() = default;

    explicit Framebuffer(u32 handle_) : handle{handle_} {}
};
} // namespace sm::gl


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_FRAMEBUFFER_IMPL)

#include "glad/glad.h"

#include "samarium/core/inline.hpp"

#include "Framebuffer.hpp"

namespace sm::gl
{
[[nodiscard]] auto Framebuffer::status() const -> u32
{
    return glCheckNamedFramebufferStatus(handle, GL_FRAMEBUFFER);
}

SM_INLINE void Framebuffer::bind() const { glBindFramebuffer(GL_FRAMEBUFFER, handle); }

SM_INLINE void Framebuffer::clear(Color color) const
{
    glClearNamedFramebufferfv(handle, GL_COLOR, 0, color.to_float_array().data());
}

SM_INLINE Framebuffer::~Framebuffer() { glDeleteFramebuffers(1, &handle); }
} // namespace sm::gl

#endif
