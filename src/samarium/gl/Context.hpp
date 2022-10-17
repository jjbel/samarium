#pragma once

#include <string> // for allocator, string

#include "glad/glad.h" // for GL_FLOAT, GL_TRUE, GL_UNSIGNED_BYTE

#include "samarium/core/types.hpp" // for f32
#include "samarium/gl/Context.hpp"
#include "samarium/util/Map.hpp" // for Map

#include "Shader.hpp"  // for Shader, FragmentShader, VertexShader
#include "Texture.hpp" // for Texture
#include "gl.hpp"      // for VertexArray, VertexAttribute, Buf...

namespace sm::gl
{
struct Context
{
    Map<std::string, VertexAttribute> attributes{
        {"position", {.size = 2, .type = GL_FLOAT, .offset = 0}},
        {"color",
         {.size = 4, .type = GL_UNSIGNED_BYTE, .offset = 2 * sizeof(f32), .normalized = GL_TRUE}},
        {"tex_coord", {.size = 2, .type = GL_FLOAT, .offset = 3 * sizeof(f32)}}};

    Map<std::string, VertexArray> vertex_arrays{};

    Map<std::string, std::string> vert_sources{};
    Map<std::string, std::string> frag_sources{};
    Map<std::string, Shader> shaders{};

    Map<std::string, Buffer<BufferType::Vertex>> vertex_buffers{};
    Map<std::string, Buffer<BufferType::Element>> element_buffers{};
    Map<std::string, Buffer<BufferType::ShaderStorage>> shader_storage_buffers{};
    Map<std::string, Texture> textures{};

    Context() = default;

    void init();

    void set_active(const Shader& shader);

  private:
    u32 active_shader_handle{};
};
} // namespace sm::gl

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_CONTEXT_IMPL)

#include "Context.hpp"

namespace sm::gl
{
void Context::init()
{
    vertex_arrays.reserve(5);
    vertex_arrays.emplace("empty", VertexArray{});
    vertex_arrays.emplace("Pos", VertexArray{{attributes.at("position")}});
    vertex_arrays.emplace("PosColor",
                          VertexArray{{attributes.at("position"), attributes.at("color")}});
    vertex_arrays.emplace("PosTex",
                          VertexArray{{attributes.at("position"), attributes.at("tex_coord")}});
    vertex_arrays.emplace("PosColorTex",
                          VertexArray{{attributes.at("position"), attributes.at("color"),
                                       attributes.at("tex_coord")}});

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

    shaders.emplace("Pos", Shader{VertexShader{vert_sources.at("Pos")},
                                  FragmentShader{frag_sources.at("Pos")}});

    shaders.emplace("PosColor", Shader{VertexShader{vert_sources.at("PosColor")},
                                       FragmentShader{frag_sources.at("PosColor")}});

    shaders.emplace("PosTex", Shader{VertexShader{vert_sources.at("PosTex")},
                                     FragmentShader{frag_sources.at("PosTex")}});

    shaders.emplace("PosColorTex", Shader{VertexShader{vert_sources.at("PosColorTex")},
                                          FragmentShader{frag_sources.at("PosColorTex")}});

    shaders.at("PosColor").bind();

    vert_sources.emplace("polyline",
#include "shaders/polyline.vert.glsl"
    );

    shaders.emplace("polyline", Shader{VertexShader{vert_sources.at("polyline")},
                                       FragmentShader{frag_sources.at("Pos")}});

    shader_storage_buffers.emplace("default", Buffer<BufferType::ShaderStorage>{});

    vertex_buffers.emplace("default", Buffer<BufferType::Vertex>{});
    element_buffers.emplace("default", Buffer<BufferType::Element>{});

    textures.emplace("default", Texture{});
}

void Context::set_active(const Shader& shader)
{
    if (shader.handle != active_shader_handle)
    {
        shader.bind();
        active_shader_handle = shader.handle;
    }
}
} // namespace sm::gl

#endif
