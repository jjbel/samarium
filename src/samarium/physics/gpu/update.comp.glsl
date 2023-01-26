R"glsl(
layout(local_size_x = 1) in;

layout(std430, binding = 2) buffer particles_buffer {
    Particle particles[];
};

void main() {
   uint index = gl_GlobalInvocationID.x; // get position to read/write data from
    particles[index].pos = vec2(index, 6969.0);
}
)glsl"
