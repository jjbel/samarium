R"glsl(
in vec2 tex_coord;
layout(binding = 0) uniform sampler2D input_texture;

out vec4 frag_color;

void main() { frag_color = texture(input_texture, tex_coord); }
)glsl"
