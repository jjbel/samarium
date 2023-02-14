R"glsl(
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 input_tex_coord;
uniform mat4 view;

out vec2 tex_coord;

void main()
{
    gl_Position  = view * vec4(position, 0.0, 1.0);
    tex_coord    = input_tex_coord;
}
)glsl"
