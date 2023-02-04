R"glsl(
#version 460

layout(std430, binding = 0) buffer ssbo { vec2 vertices[]; };

uniform mat4 view;
uniform vec2 screen_dims;
uniform float thickness;

void main()
{
    int line_index = gl_VertexID / 6;
    int tri_index  = gl_VertexID % 6;

    vec4 va[4];
    for (int i = 0; i < 4; ++i)
    {
        vec4 pos = view * vec4(vertices[line_index + i], 0.0, 1.0);
        pos.xy   = (pos.xy + 1.0) * 0.5 * screen_dims;
        va[i]    = pos;
    }

    vec2 v_line  = normalize(va[2].xy - va[1].xy);
    vec2 nv_line = vec2(-v_line.y, v_line.x);

    vec4 pos;
    if (tri_index == 0 || tri_index == 1 || tri_index == 3)
    {
        vec2 v_pred  = normalize(va[1].xy - va[0].xy);
        vec2 v_miter = normalize(nv_line + vec2(-v_pred.y, v_pred.x));

        pos = va[1];
        pos.xy += v_miter * thickness * (tri_index == 1 ? -0.5 : 0.5) / dot(v_miter, nv_line);
    }
    else
    {
        vec2 v_succ  = normalize(va[3].xy - va[2].xy);
        vec2 v_miter = normalize(nv_line + vec2(-v_succ.y, v_succ.x));

        pos = va[2];
        pos.xy += v_miter * thickness * (tri_index == 5 ? 0.5 : -0.5) / dot(v_miter, nv_line);
    }

    pos.xy = pos.xy / screen_dims * 2.0 - 1.0;
    pos.xyz *= pos.w;
    gl_Position = pos;
}
)glsl"
