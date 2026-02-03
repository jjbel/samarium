R"glsl(
in vec2 tex_coord;

uniform sampler2D input_texture;
uniform vec4 color;

out vec4 frag_color;

void main()
{
    // the texture is only red channel, use it to make an alpha map
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(input_texture, tex_coord));
    frag_color        = color * sampled;

    // use this to view just the bounding boxes of the chars
    // frag_color     = color;
}
)glsl"
