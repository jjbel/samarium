/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <string> // for string

#include "glad/glad.h" // for GL_FLOAT, GL_TRUE, GL_UNSIGNED_BYTE

#include "samarium/core/types.hpp"       // for f32
#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/util/unordered.hpp"   // for Map

#include "Framebuffer.hpp" // for Framebuffer
#include "Shader.hpp"      // for Shader, FragmentShader, VertexShader
#include "Texture.hpp"     // for Texture
#include "Vertex.hpp"      // for Vertex
#include "gl.hpp"          // for VertexArray, VertexAttribute, Buf...

namespace sm::gl
{
struct Context
{
    Map<std::string, VertexAttribute> attributes{
        {"position", {.size = 2, .type = GL_FLOAT, .offset = 0}},
        {"color",
         {.size = 4, .type = GL_UNSIGNED_BYTE, .offset = 2 * sizeof(f32), .normalized = GL_TRUE}},
        {"PosTex.tex_coord", {.size = 2, .type = GL_FLOAT, .offset = 2 * sizeof(f32)}},
        {"PosColorTex.tex_coord", {.size = 2, .type = GL_FLOAT, .offset = 3 * sizeof(f32)}}};

    Map<std::string, VertexArray> vertex_arrays{};

    Map<std::string, std::string> vert_sources{};
    Map<std::string, std::string> frag_sources{};
    Map<std::string, Shader> shaders{};

    Map<std::string, VertexBuffer> vertex_buffers{};
    Map<std::string, ElementBuffer> element_buffers{};
    Map<std::string, ShaderStorageBuffer> shader_storage_buffers{};
    Map<std::string, Texture> textures{};

    Texture frame_texture;
    Framebuffer framebuffer;

    explicit Context(Dimensions dims);

    void set_active(const Shader& shader);

    void set_active(const VertexArray& vertex_array);

    void draw_frame();

  private:
    u32 active_shader_handle{};
    u32 active_vertex_array_handle{};
};
} // namespace sm::gl

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_CONTEXT_IMPL)

#include "samarium/core/inline.hpp"

#include "Context.hpp"

namespace sm::gl
{
SM_INLINE Context::Context(Dimensions dims)
    : frame_texture{ImageFormat::RGBA8, dims, Texture::Wrap::ClampEdge, Texture::Filter::Nearest,
                    Texture::Filter::Nearest},
      framebuffer(frame_texture)
{
    vertex_arrays.reserve(5);
    vertex_arrays.emplace("empty", VertexArray{});
    vertex_arrays.emplace("Pos", VertexArray{{attributes.at("position")}});
    vertex_arrays.emplace("PosColor",
                          VertexArray{{attributes.at("position"), attributes.at("color")}});
    vertex_arrays.emplace(
        "PosTex", VertexArray{{attributes.at("position"), attributes.at("PosTex.tex_coord")}});
    vertex_arrays.emplace("PosColorTex",
                          VertexArray{{attributes.at("position"), attributes.at("color"),
                                       attributes.at("PosColorTex.tex_coord")}});

    vert_sources.emplace("Pos",
#include "shaders/Pos.vert.glsl"
    );
    vert_sources.emplace("PosColor",
#include "shaders/PosColor.vert.glsl"
    );
    vert_sources.emplace("PosTex",
#include "shaders/PosTex.vert.glsl"
    );
    vert_sources.emplace("PosColorTex",
#include "shaders/PosColorTex.vert.glsl"
    );

    frag_sources.emplace("Pos",
#include "shaders/Pos.frag.glsl"
    );
    frag_sources.emplace("PosColor",
#include "shaders/PosColor.frag.glsl"
    );
    frag_sources.emplace("PosTex",
#include "shaders/PosTex.frag.glsl"
    );
    frag_sources.emplace("PosColorTex",
#include "shaders/PosColorTex.frag.glsl"
    );
    frag_sources.emplace("text",
#include "shaders/text.frag.glsl"
    );

    shaders.emplace("Pos", Shader{expect(VertexShader::make(vert_sources.at("Pos"))),
                                  expect(FragmentShader::make(frag_sources.at("Pos")))});

    shaders.emplace("PosColor", Shader{expect(VertexShader::make(vert_sources.at("PosColor"))),
                                       expect(FragmentShader::make(frag_sources.at("PosColor")))});

    shaders.emplace("PosTex", Shader{expect(VertexShader::make(vert_sources.at("PosTex"))),
                                     expect(FragmentShader::make(frag_sources.at("PosTex")))});

    shaders.emplace("PosColorTex",
                    Shader{expect(VertexShader::make(vert_sources.at("PosColorTex"))),
                           expect(FragmentShader::make(frag_sources.at("PosColorTex")))});

    shaders.emplace("text", Shader{expect(VertexShader::make(vert_sources.at("PosTex"))),
                                   expect(FragmentShader::make(frag_sources.at("text")))});

    vert_sources.emplace("polyline",
#include "shaders/polyline.vert.glsl"
    );

    shaders.emplace("polyline", Shader{*VertexShader::make(vert_sources.at("polyline")),
                                       *FragmentShader::make(frag_sources.at("Pos"))});

    shader_storage_buffers.emplace("default", ShaderStorageBuffer{});

    vertex_buffers.emplace("default", VertexBuffer{});
    element_buffers.emplace("default", ElementBuffer{});

    //    textures.emplace("default", Texture{});

    shaders.at("Pos").bind();
    vertex_arrays.at("Pos").bind();
    framebuffer.bind();
}

SM_INLINE void Context::set_active(const Shader& shader)
{
    if (shader.handle != active_shader_handle)
    {
        shader.bind();
        active_shader_handle = shader.handle;
    }
}

SM_INLINE void Context::set_active(const VertexArray& vertex_array)
{
    if (vertex_array.handle != active_vertex_array_handle)
    {
        vertex_array.bind();
        active_vertex_array_handle = vertex_array.handle;
    }
}

SM_INLINE void Context::draw_frame()
{
    using Vert                        = Vertex<Layout::PosTex>;
    static constexpr auto buffer_data = std::to_array<Vert>({{{-1, -1}, {0, 0}},
                                                             {{1, 1}, {1, 1}},
                                                             {{-1, 1}, {0, 1}},

                                                             {{-1, -1}, {0, 0}},
                                                             {{1, -1}, {1, 0}},
                                                             {{1, 1}, {1, 1}}});

    framebuffer.unbind();
    const auto& shader = shaders.at("PosTex");
    set_active(shader);
    shader.set("view", glm::mat4{1.0F});

    frame_texture.bind();

    auto& vao = vertex_arrays.at("PosTex");
    set_active(vao);

    const auto& buffer = vertex_buffers.at("default");
    buffer.set_data(buffer_data);
    vao.bind(buffer, sizeof(Vert));

    glDrawArrays(GL_TRIANGLES, 0, static_cast<i32>(buffer_data.size()));

    framebuffer.bind();
}
} // namespace sm::gl

#endif
