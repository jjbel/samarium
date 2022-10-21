/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "glad/glad.h" // for GLenum, GL_RGBA8, glDeleteTex...

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


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TEXTURE_IMPL)

#include "glad/glad.h" // for GLenum, GLint, glTextureParameteri, GL_FALSE

#include "samarium/core/inline.hpp"

#include "Texture.hpp"

namespace sm::gl
{
SM_INLINE Texture::Texture(WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
{
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    // set the texture wrapping parameters
    glTextureParameteri(handle, GL_TEXTURE_WRAP_S, static_cast<GLint>(mode));
    glTextureParameteri(handle, GL_TEXTURE_WRAP_T, static_cast<GLint>(mode));

    // set texture filtering parameters
    glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
    glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
}

SM_INLINE
Texture::Texture(const Image& image, WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    set_data(image);
}

SM_INLINE Texture::Texture(
    Dimensions dims, GLenum type, WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    create(dims, type);
}

SM_INLINE void Texture::create(Dimensions dims, GLenum type)
{
    const auto width  = static_cast<i32>(dims.x);
    const auto height = static_cast<i32>(dims.y);
    glTextureStorage2D(handle, 1, type, width, height);
}

SM_INLINE void Texture::set_data(const Image& image)
{
    const auto width  = static_cast<i32>(image.dims.x);
    const auto height = static_cast<i32>(image.dims.y);
    create(image.dims, GL_RGBA8);
    // load the image data
    glTextureSubImage2D(handle, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                        static_cast<const void*>(&image.front()));

    glGenerateTextureMipmap(handle);
}

SM_INLINE void Texture::bind(u32 texture_unit_index)
{
    glBindTextureUnit(texture_unit_index, handle);
}

SM_INLINE void Texture::bind_level(u32 texture_unit_index, i32 level, Access access, GLenum format)
{
    glBindImageTexture(texture_unit_index, handle, level, GL_FALSE, 0, static_cast<GLenum>(access),
                       format);
}
} // namespace sm::gl

#endif
