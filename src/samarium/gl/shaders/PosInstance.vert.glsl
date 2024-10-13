R"glsl(
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 instance_position;

uniform vec4 view;

void main()
{
    gl_Position  = vec4(view.xy + view.zw * (instance_position + position), 0.0, 1.0);
}
)glsl"
