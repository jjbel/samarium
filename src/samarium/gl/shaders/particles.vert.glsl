R"glsl(
layout(std430, binding = 2) buffer ssbo { Particle data[]; };

layout(location = 0) in vec2 position;
uniform float scale;
uniform mat4 view;

void main()
{
    Particle particle = data[gl_InstanceID];
    vec2 pos = position * particle.radius * scale + particle.pos;
    gl_Position       = view * vec4(pos, 0.0, 1.0);
}
)glsl"
