R"glsl(
#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 input_color;
layout(location = 2) in vec2 input_tex_coord;
uniform mat4 view;

out vec4 vertex_color;
out vec2 tex_coord;

void main()
{
    gl_Position  = view * vec4(position, 0.0, 1.0);
    vertex_color = input_color;
    tex_coord    = input_tex_coord;
}
)glsl"