R"glsl(
layout(location = 0) in vec2 position;
uniform mat4 view;

void main()
{
    gl_Position  = view * vec4(position, 0.0, 1.0);
}
)glsl"
