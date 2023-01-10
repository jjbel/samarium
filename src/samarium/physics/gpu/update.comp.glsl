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
//    particle.vel += particle.acc * delta_time;
//    particle.pos += particle.vel * delta_time;
//    particles[pos] = particle;

    particles[pos].pos = vec2(2.0, 3.0);
}
)glsl"
