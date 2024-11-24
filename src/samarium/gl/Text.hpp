/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <filesystem>  // for path
#include <span>        // for span
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <string_view> // for string_view


#include "glad/glad.h"

#include "ft2build.h"
#include FT_FREETYPE_H

#include "samarium/core/types.hpp"     // for i32
#include "samarium/gl/Context.hpp"     // for Context
#include "samarium/gl/Texture.hpp"     // for Texture
#include "samarium/gl/Vertex.hpp"      // for Vertex
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/Vec2.hpp"      // for Vec2_t
#include "samarium/util/Result.hpp"    // for Result
#include "samarium/util/unordered.hpp" // for Map

namespace sm
{
inline Result<std::filesystem::path> find_fonts(const std::vector<std::string>& fonts)
{
    // TODO some are in C:|Users\user\AppData\Microsoft\Windows\Fonts, but how to get user?
    const std::vector<std::string> default_dirs = {"C:\\Windows\\Fonts", "~/.local/share/fonts",
                                                   "/usr/local/share/fonts", " /usr/share/fonts"};

    for (const auto& dir : default_dirs)
    {
        for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(dir))
        {
            if (!dir_entry.is_regular_file()) { continue; }
            for (auto font : fonts)
            {
                if (dir_entry.path().stem().string() == font) // TODO tolower?
                {
                    print(dir_entry.path().string());
                    return {dir_entry.path()};
                }
            }
        }
    }
    return make_unexpected("Error: font not found.");
}

inline auto find_font()
{
    const std::vector<std::string> default_fonts = {"Arial", "CascadiaCode", "UbuntuMono",
                                                    "Ubuntu"};
    return expect(find_fonts(default_fonts));
}

inline auto find_font(const std::string& font)
{
    const auto path = std::filesystem::path(font);
    if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) { return path; }
    return expect(find_fonts({font}));
}
} // namespace sm

// TODO file in gl but namespace is draw
namespace sm::draw
{
/// Holds all state information relevant to a character as loaded using FreeType
struct Character
{
    gl::Texture texture; // glyph texture
    Vec2_t<u32> size;    // Size of glyph
    Vec2_t<i32> bearing; // Offset from baseline to left/top of glyph
    f32 advance;         // Horizontal offset to advance to next glyph
};


// TODO handle newlines?
// https://learnopengl.com/In-Practice/Text-Rendering
struct Text
{
    u32 pixel_height{};
    Map<char, Character> characters{};

    [[nodiscard]] static auto make(std::string font = "", u32 pixel_height = 96) -> Result<Text>
    {
        const auto font_path = find_font(font).string();
        auto* ft             = FT_Library{};

        // All functions return a value different than 0 whenever an error occurred
        if (FT_Init_FreeType(&ft) != 0)
        {
            return make_unexpected(std::string{"Could not initialize FreeType"});
        }

        // load font as face
        auto* face = FT_Face{};

        if (FT_New_Face(ft, font_path.c_str(), 0, &face) != 0)
        {
            return make_unexpected(fmt::format("Could not create font: {}", font_path));
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
                    fmt::format("Could not create glyph for {} for font: {}", c, font_path));
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
                              static_cast<f32>(face->glyph->advance.x) / 64.0F}});
            // advance is number of 1/64 pixels
            // could use bitshift instead
        }


        // destroy FreeType once we're finished
        FT_Done_Face(face);
        FT_Done_FreeType(ft);

        return text;
    }

    // returns in pixels, ie not divided by pixel_height
    BoundingBox<f64> bounding_box(const Character& c, f64 existing_advance = 0) const
    {
        return {{existing_advance, static_cast<f64>(c.bearing.y) - static_cast<f64>(c.size.y)},
                {existing_advance + static_cast<f64>(c.advance), static_cast<f64>(c.bearing.y)}};
    }

    BoundingBox<f64> bounding_box(const std::string& text, f64 scale = 1.0) const
    {
        auto current_advance = f32{};
        auto box             = BoundingBox<f64>{};
        bool first           = true;

        for (auto c : text)
        {
            const auto& ch    = characters.at(c);
            const auto ch_box = bounding_box(ch, current_advance);
            if (first)
            {
                box   = ch_box;
                first = false;
            }
            else if (ch.size.x * ch.size.y != 0) { box = BoundingBox<f64>::fit_boxes(box, ch_box); }

            current_advance += ch.advance;
        }

        scale /= static_cast<f32>(pixel_height);
        box.min *= scale;
        box.max *= scale;
        return box;
    }

    Vec2f placement_movement(const std::string& text, f32 scale, Placement p) const
    {
        return bounding_box(text, scale).get_placement(p).cast<f32>();
    }

    BoundingBox<f64> bounding_box(const std::string& text, f64 scale, Placement p) const
    {
        auto box         = bounding_box(text, scale);
        const auto delta = -box.get_placement(p);
        box.min += delta;
        box.max += delta;
        return box;
    }


    void operator()(gl::Context& context,
                    const std::string& text,
                    Vec2f pos,
                    f32 scale,
                    Color color,
                    const glm::mat4& transform,
                    Placement p = {PlacementX::Middle, PlacementY::Middle});

    // scale is in world space
    void operator()(Window& window,
                    const std::string& text,
                    Vec2f pos   = {},
                    f32 scale   = 1.0,
                    Color color = Color{255, 255, 255},
                    Placement p = {PlacementX::Middle, PlacementY::Middle});
};
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TEXT_IMPL)

#include "samarium/core/inline.hpp"

namespace sm::draw
{
SM_INLINE void Text::operator()(gl::Context& context,
                                const std::string& text,
                                Vec2f pos,
                                f32 scale,
                                Color color,
                                const glm::mat4& transform,
                                Placement p)
{
    // TODO add origin point: 9 possible. or at least the common ones. maybe make it x and y
    // separate
    pos -= placement_movement(text, scale, p);

    // divide by pixel_height
    // ch.size.y was from 0 to pixel_height, now its from 0 to scale, in world space
    // TODO draw a circle next to it: not EXACTLY of size "scale", but seems to be bigger
    // TODO this shd be done in make(), not in each function individually
    scale /= static_cast<f32>(pixel_height);

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
            pos.x += ch.advance * scale;
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

        pos.x += ch.advance * scale;
    }
}

SM_INLINE void Text::operator()(
    Window& window, const std::string& text, Vec2f pos, f32 scale, Color color, Placement p)
{
    this->operator()(window.context, text, pos, scale, color, window.world2gl(), p);
}
} // namespace sm::draw

#endif
