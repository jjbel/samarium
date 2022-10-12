#include <glad/glad.h> // for GLenum, GLint, glTextureParameteri, GL_FALSE

#include "Texture.hpp"

namespace sm::gl
{
Texture::Texture(WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
{
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    // set the texture wrapping parameters
    glTextureParameteri(handle, GL_TEXTURE_WRAP_S, static_cast<GLint>(mode));
    glTextureParameteri(handle, GL_TEXTURE_WRAP_T, static_cast<GLint>(mode));

    // set texture filtering parameters
    glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
    glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag_filter));
}

Texture::Texture(const Image& image, WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    set_data(image);
}

Texture::Texture(
    Dimensions dims, GLenum type, WrapMode mode, FilterMode min_filter, FilterMode mag_filter)
    : Texture(mode, min_filter, mag_filter)
{
    create(dims, type);
}

void Texture::create(Dimensions dims, GLenum type)
{
    const auto width  = static_cast<i32>(dims.x);
    const auto height = static_cast<i32>(dims.y);
    glTextureStorage2D(handle, 1, type, width, height);
}

void Texture::set_data(const Image& image)
{
    const auto width  = static_cast<i32>(image.dims.x);
    const auto height = static_cast<i32>(image.dims.y);
    create(image.dims, GL_RGBA8);
    // load the image data
    glTextureSubImage2D(handle, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                        static_cast<const void*>(&image.front()));

    glGenerateTextureMipmap(handle);
}

void Texture::bind(u32 texture_unit_index) { glBindTextureUnit(texture_unit_index, handle); }

void Texture::bind_level(u32 texture_unit_index, i32 level, Access access, GLenum format)
{
    glBindImageTexture(texture_unit_index, handle, level, GL_FALSE, 0, static_cast<GLenum>(access),
                       format);
}
} // namespace sm::gl
