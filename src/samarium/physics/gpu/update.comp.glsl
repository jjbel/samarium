R"glsl(
layout(local_size_x = 1) in;

layout(std430, binding = 0) buffer particles_buffer {
    Particle particles[];
};

// layout(std430, binding = 1) buffer delta_time_buffer {
//     float delta_time;
// };

void main() {
   uint pos = gl_GlobalInvocationID.x; // get position to read/write data from
//    Particle particle = Particle(particles[pos]);
//    particle.vel += particle.acc;
//    particle.pos += particle.vel;
//    particles[pos] = particle;
    particles[pos].pos = vec2(122.0, 123.0);
}
)glsl"
