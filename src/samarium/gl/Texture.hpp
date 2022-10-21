/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "glad/glad.h"                 // for GLenum, GL_RGBA8, glDeleteTex...

#include "samarium/core/types.hpp"     // for u32, i32
#include "samarium/graphics/Image.hpp" // for Image
#include "samarium/math/Vector2.hpp"   // for Dimensions

namespace sm::gl
{
enum class Access
{
    Read      = GL_READ_ONLY,
    Write     = GL_WRITE_ONLY,
    ReadWrite = GL_READ_WRITE
};

struct Texture
{
    enum class WrapMode
    {
        Repeat      = GL_REPEAT,
        Mirror      = GL_MIRRORED_REPEAT,
        ClampEdge   = GL_CLAMP_TO_EDGE,
        ClampBorder = GL_CLAMP_TO_BORDER
    };

    enum class FilterMode
    {
        Linear               = GL_LINEAR,
        Nearest              = GL_NEAREST,
        LinearMipmapLinear   = GL_LINEAR_MIPMAP_LINEAR,
        LinearMipmapNearest  = GL_LINEAR_MIPMAP_NEAREST,
        NearestMipmapLinear  = GL_NEAREST_MIPMAP_LINEAR,
        NearestMipmapNearest = GL_NEAREST_MIPMAP_NEAREST
    };

    u32 handle;

    explicit Texture(WrapMode mode         = WrapMode::Repeat,
                     FilterMode min_filter = FilterMode::LinearMipmapLinear,
                     FilterMode mag_filter = FilterMode::Linear);

    explicit Texture(const Image& image,
                     WrapMode mode         = WrapMode::Repeat,
                     FilterMode min_filter = FilterMode::LinearMipmapLinear,
                     FilterMode mag_filter = FilterMode::Linear);

    explicit Texture(Dimensions dims,
                     GLenum type,
                     WrapMode mode         = WrapMode::Repeat,
                     FilterMode min_filter = FilterMode::LinearMipmapLinear,
                     FilterMode mag_filter = FilterMode::Linear);

    void create(Dimensions dims, GLenum type = GL_RGBA8);

    void set_data(const Image& image);

    void bind(u32 texture_unit_index = 0U);

    void bind_level(u32 texture_unit_index = 0,
                    i32 level              = 0,
                    Access access          = Access::ReadWrite,
                    GLenum format          = GL_RGBA8);

    Texture(const Texture&) = delete;

    Texture(Texture&& other) noexcept : handle{other.handle} { other.handle = 0; }

    Texture& operator=(Texture&& other) noexcept
    {
        if (this != &other)
        {
            glDeleteTextures(1, &handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    ~Texture() { glDeleteTextures(1, &handle); }
};
} // namespace sm::gl
