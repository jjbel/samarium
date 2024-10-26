/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <filesystem> // for path
#include <string>     // for string

#include "glad/glad.h"

#include "ft2build.h"
#include FT_FREETYPE_H

#include "samarium/core/types.hpp"     // for i32
#include "samarium/gl/Context.hpp"     // for Context
#include "samarium/gl/Texture.hpp"     // for Texture
#include "samarium/gl/Vertex.hpp"      // for Vertex
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/Vector2.hpp"   // for Vector2_t
#include "samarium/util/Result.hpp"    // for Result
#include "samarium/util/unordered.hpp" // for Map

// TODO file in gl but namespace is draw
namespace sm::draw
{
/// Holds all state information relevant to a character as loaded using FreeType
struct Character
{
    gl::Texture texture;    // glyph texture
    Vector2_t<u32> size;    // Size of glyph
    Vector2_t<i32> bearing; // Offset from baseline to left/top of glyph
    u32 advance;            // Horizontal offset to advance to next glyph
};

struct Text
{
    u32 pixel_height{};
    Map<char, Character> characters{};

    static Result<std::filesystem::path> default_font_dir()
    {
#if defined(_WIN32)
        const auto path = std::filesystem::path("C:\\Windows\\Fonts\\");
#elif defined(linux)
        // TODO make it a priority list: /usr/local/share/fonts/ also
        // maybe let user add font paths
        const auto path = std::filesystem::path("/usr/share/fonts/");
#else
#error "unknown platform"
#endif
        if (std::filesystem::exists(path)) { return {path}; }
        return make_unexpected("Default font path not found");
    }

    [[nodiscard]] static auto make(std::string font = "arial.ttf",
                                   u32 pixel_height = 96) -> Result<Text>
    {
        const auto font_dir = default_font_dir();
        // TODO more idiomatic way
        if (font_dir && std::filesystem::exists(font_dir.value() / font))
        {
            font = (font_dir.value() / font).lexically_normal().string();
        }
        else
        {
            if (!std::filesystem::exists(font))
            {
                return make_unexpected(fmt::format("{} does not exist", font));
            }

            if (!std::filesystem::is_regular_file(font))
            {
                return make_unexpected(fmt::format("{} is not a file", font));
            }
        }

        auto* ft = FT_Library{};

        // All functions return a value different than 0 whenever an error occurred
        if (FT_Init_FreeType(&ft) != 0)
        {
            return make_unexpected(std::string{"Could not initialize FreeType"});
        }

        // load font as face
        auto* face = FT_Face{};

        if (FT_New_Face(ft, font.c_str(), 0, &face) != 0)
        {
            return make_unexpected(fmt::format("Could not create font: {}", font));
        }

        // set size to load glyphs as
        FT_Set_Pixel_Sizes(face, 0, pixel_height);


        // TODO doesn't do anything?
        // disable byte-alignment restriction
        // https://stackoverflow.com/a/58927549
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        auto text = Text{pixel_height};

        // load first 128 characters of ASCII set
        for (/* unsigned */ char c = ' '; c <= '~'; c++)
        {
            // Load character glyph
            if (FT_Load_Char(face, static_cast<u64>(c), FT_LOAD_RENDER) != 0)
            {
                return make_unexpected(
                    fmt::format("Could not create glyph for {} for font: {}", c, font));
            }

            const auto& bitmap = face->glyph->bitmap;
            auto texture       = gl::Texture(gl::ImageFormat::R8, gl::Texture::Wrap::ClampEdge,
                                             gl::Texture::Filter::Linear, gl::Texture::Filter::Linear);

            // some characters eg space don't have data but take up space
            if (bitmap.width * bitmap.rows != 0)
            {
                texture.set_data(
                    std::span{bitmap.buffer, static_cast<u64>(bitmap.width * bitmap.rows)},
                    {static_cast<u64>(bitmap.width), static_cast<u64>(bitmap.rows)});
            }

            text.characters.insert(
                {c, Character{std::move(texture),
                              {bitmap.width, bitmap.rows},
                              {face->glyph->bitmap_left, face->glyph->bitmap_top},
                              static_cast<u32>(face->glyph->advance.x)}});
        }


        // destroy FreeType once we're finished
        FT_Done_Face(face);
        FT_Done_FreeType(ft);

        return text;
    }

    void operator()(gl::Context& context,
                    const std::string& text,
                    Vector2f pos,
                    f32 scale,
                    Color color,
                    const glm::mat4& transform);

    // scale is in world space
    void operator()(Window& window,
                    const std::string& text,
                    Vector2f pos = {},
                    f32 scale    = 1.0,
                    Color color  = Color{255, 255, 255});
};
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TEXT_IMPL)

#include "samarium/core/inline.hpp"

namespace sm::draw
{
SM_INLINE void Text::operator()(gl::Context& context,
                                const std::string& text,
                                Vector2f pos,
                                f32 scale,
                                Color color,
                                const glm::mat4& transform)
{
    // TODO add origin point: 9 possible. or at least the common ones. maybe make it x and y
    // separate

    const auto& shader = context.shaders.at("text");
    context.set_active(shader);
    shader.set("view", static_cast<glm::mat4>(transform));
    shader.set("color", color);
    // TODO cud simplify calcn by modifying transform using scale

    // iterate through all characters
    for (auto c : text)
    {
        auto& ch = characters.at(c);
        if (ch.size.x * ch.size.y == 0)
        {
            // now advance cursors for next glyph
            // advance is number of 1/64 pixels
            // could use bitshift instead
            pos.x += static_cast<f32>(ch.advance) / 64.0F * scale;
            continue;
        }

        const auto xpos = pos.x + static_cast<f32>(ch.bearing.x) * scale;
        const auto ypos = pos.y - static_cast<f32>(ch.size.y - ch.bearing.y) * scale;

        const auto w = static_cast<f32>(ch.size.x) * scale;
        const auto h = static_cast<f32>(ch.size.y) * scale;
        // update VBO for each character
        const f32 vertices[6][4] = {{xpos, ypos + h, 0.0F, 0.0F},    {xpos, ypos, 0.0F, 1.0F},
                                    {xpos + w, ypos, 1.0F, 1.0F},

                                    {xpos, ypos + h, 0.0F, 0.0F},    {xpos + w, ypos, 1.0F, 1.0F},
                                    {xpos + w, ypos + h, 1.0F, 0.0F}};
        ch.texture.bind();

        auto& vao = context.vertex_arrays.at("PosTex");
        context.set_active(vao);
        // TODO use a dedicated buffer for this
        const auto& buffer = context.vertex_buffers.at("default");
        buffer.set_data(vertices);
        vao.bind(buffer, sizeof(gl::Vertex<gl::Layout::PosTex>));

        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

        pos.x += static_cast<f32>(ch.advance) / 64.0F * scale;
    }
}

SM_INLINE void
Text::operator()(Window& window, const std::string& text, Vector2f pos, f32 scale, Color color)
{
    // divide by pixel_height
    // ch.size.y was from 0 to pixel_height, now its from 0 to scale, in world space
    // TODO draw a circle next to it: not EXACTLY of size "scale", but seems to be bigger
    this->operator()(window.context, text, pos, scale / static_cast<f32>(pixel_height), color,
                     window.world2gl());
}
} // namespace sm::draw

#endif
