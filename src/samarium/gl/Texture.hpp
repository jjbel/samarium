/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "glad/glad.h"                   // for GLenum, GL_RGBA8, glDeleteTex...
#include "range/v3/range/primitives.hpp" // for data

#include "samarium/core/types.hpp"     // for u32, i32
#include "samarium/graphics/Image.hpp" // for Image
#include "samarium/math/Vector2.hpp"   // for Dimensions

#include "gl.hpp" // for Access

namespace sm::gl
{
struct Texture
{
    enum class Wrap
    {
        Repeat      = GL_REPEAT,
        Mirror      = GL_MIRRORED_REPEAT,
        ClampEdge   = GL_CLAMP_TO_EDGE,
        ClampBorder = GL_CLAMP_TO_BORDER
    };

    enum class Filter
    {
        Linear               = GL_LINEAR,
        Nearest              = GL_NEAREST,
        LinearMipmapLinear   = GL_LINEAR_MIPMAP_LINEAR,
        LinearMipmapNearest  = GL_LINEAR_MIPMAP_NEAREST,
        NearestMipmapLinear  = GL_NEAREST_MIPMAP_LINEAR,
        NearestMipmapNearest = GL_NEAREST_MIPMAP_NEAREST
    };

    u32 handle;

    explicit Texture(Wrap mode         = Wrap::Repeat,
                     Filter min_filter = Filter::LinearMipmapLinear,
                     Filter mag_filter = Filter::Linear);

    explicit Texture(const Image& image,
                     Wrap mode         = Wrap::Repeat,
                     Filter min_filter = Filter::LinearMipmapLinear,
                     Filter mag_filter = Filter::Linear);

    explicit Texture(std::span<Color> image,
                     Wrap mode         = Wrap::Repeat,
                     Filter min_filter = Filter::LinearMipmapLinear,
                     Filter mag_filter = Filter::Linear);

    explicit Texture(Dimensions dims,
                     Wrap mode         = Wrap::Repeat,
                     Filter min_filter = Filter::LinearMipmapLinear,
                     Filter mag_filter = Filter::Linear);

    void set_data(const Image& image, bool mipmaps = true);

    void set_data(const ranges::range auto& data,
                  Dimensions dims,
                  GLenum internal_format,
                  GLenum format,
                  GLenum datatype)
    {
        create(dims, internal_format);

        glTextureSubImage2D(handle, 0, 0, 0, static_cast<i32>(dims.x), static_cast<i32>(dims.y),
                            format, datatype, static_cast<const void*>(ranges::data(data)));
    }

    void bind(u32 texture_unit_index = 0U);

    void bind_level(u32 texture_unit_index = 0,
                    i32 level              = 0,
                    Access access          = Access::ReadWrite,
                    GLenum format          = GL_RGBA8);

    void make_mipmaps();

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

  private:
    void create(Dimensions dims, GLenum internal_format = GL_RGBA8);
};
} // namespace sm::gl


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TEXTURE_IMPL)

#include "glad/glad.h" // for GLenum, GLint, glTextureParameteri, GL_FALSE

#include "samarium/core/inline.hpp"

#include "Texture.hpp"

namespace sm::gl
{
SM_INLINE Texture::Texture(Wrap mode, Filter min_filter, Filter mag_filter)
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
Texture::Texture(const Image& image, Wrap mode, Filter min_filter, Filter mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    set_data(image);
}

SM_INLINE
Texture::Texture(Dimensions dims, Wrap mode, Filter min_filter, Filter mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    create(dims);
}

SM_INLINE void Texture::make_mipmaps() { glGenerateTextureMipmap(handle); }

SM_INLINE void Texture::create(Dimensions dims, GLenum internal_format)
{
    glTextureStorage2D(handle, 1, internal_format, static_cast<i32>(dims.x),
                       static_cast<i32>(dims.y));
}

SM_INLINE void Texture::set_data(const Image& image, bool mipmaps)
{
    create(image.dims);
    glTextureSubImage2D(handle, 0, 0, 0, static_cast<i32>(image.dims.x),
                        static_cast<i32>(image.dims.y), GL_RGBA, GL_UNSIGNED_BYTE,
                        static_cast<const void*>(image.data()));
    if (mipmaps) { make_mipmaps(); }
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
