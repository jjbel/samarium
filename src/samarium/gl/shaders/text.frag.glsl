R"glsl(
#version 460 core

in vec2 tex_coord;

uniform sampler2D input_texture;
uniform vec4 color;

out vec4 frag_color;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(input_texture, tex_coord).r);
    frag_color        = color * sampled;
}
)glsl"
