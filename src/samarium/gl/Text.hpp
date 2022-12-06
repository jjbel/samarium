/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
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
    Map<char, Character> characters{};

    [[nodiscard]] static auto make(const std::filesystem::path& font_path, u32 height = 48)
        -> Result<Text>
    {
        if (!std::filesystem::exists(font_path))
        {
            return tl::make_unexpected(fmt::format("{} does not exist", font_path));
        }

        if (!std::filesystem::is_regular_file(font_path))
        {
            return tl::make_unexpected(fmt::format("{} is not a file", font_path));
        }

        auto* ft = FT_Library{};

        // All functions return a value different than 0 whenever an error occurred
        if (FT_Init_FreeType(&ft) != 0)
        {
            return tl::make_unexpected(std::string{"Could not initialize FreeType"});
        }

        // load font as face
        auto* face = FT_Face{};

        if (FT_New_Face(ft, font_path.string().c_str(), 0, &face) != 0)
        {
            return tl::make_unexpected(fmt::format("Could not create font: {}", font_path));
        }

        // set size to load glyphs as
        FT_Set_Pixel_Sizes(face, 0, height);

        // disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        auto text = Text{};

        // load first 128 characters of ASCII set
        for (/* unsigned */ char c = ' '; c <= '~'; c++)
        {
            // Load character glyph
            if (FT_Load_Char(face, static_cast<u64>(c), FT_LOAD_RENDER) != 0)
            {
                return tl::make_unexpected(
                    fmt::format("Could not create glyph for {} for font: {}", c, font_path));
            }

            const auto& bitmap = face->glyph->bitmap;
            // fmt::print("\nglyph for ({}) width: {} rows: {}. ", c, bitmap.width, bitmap.rows);
            auto texture = gl::Texture{gl::Texture::Wrap::ClampEdge, gl::Texture::Filter::Linear,
                                       gl::Texture::Filter::Linear};

            // some characters eg space don't have data but take up space
            if (bitmap.width * bitmap.rows != 0)
            {
                texture.set_data(
                    std::span{bitmap.buffer, static_cast<u64>(bitmap.width * bitmap.rows)},
                    {static_cast<u64>(bitmap.width), static_cast<u64>(bitmap.rows)}, GL_R8, GL_RED,
                    GL_UNSIGNED_BYTE);
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
                    glm::mat4 transform);

    void operator()(Window& window, const std::string& text, Vector2f pos, f32 scale, Color color);
};
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"

namespace sm::draw
{
SM_INLINE void Text::operator()(gl::Context& context,
                                const std::string& text,
                                Vector2f pos,
                                f32 scale,
                                Color color,
                                glm::mat4 transform)
{
    scale /= 1000.0F; // FIXME pixel to screen size

    const auto& shader = context.shaders.at("text");
    context.set_active(shader);
    shader.set("view", transform);
    shader.set("color", color);

    // iterate through all characters
    for (auto c : text)
    {
        auto& ch = characters.at(c);
        if (ch.size.x * ch.size.y == 0)
        {
            pos.x += static_cast<f32>(ch.advance >> 6) * scale;
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
        const auto& buffer = context.vertex_buffers.at("default");
        buffer.set_data(vertices);
        vao.bind(buffer, sizeof(gl::Vertex<gl::Layout::PosTex>));

        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        pos.x +=
            (ch.advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64 (divide
                                       // amount of 1/64th pixels by 64 to get amount of pixels))
    }
}

SM_INLINE void
Text::operator()(Window& window, const std::string& text, Vector2f pos, f32 scale, Color color)
{
    this->operator()(window.context, text, pos, scale, color, window.view);
}
} // namespace sm::draw

#endif
