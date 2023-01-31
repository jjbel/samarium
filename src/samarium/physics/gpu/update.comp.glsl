R"glsl(
layout(std430, binding = 2) buffer ssbo {
  Particle data[];
};

layout(location = 0) uniform float delta_time;

void update(inout Particle particle) {
    particle.vel += particle.acc * delta_time;
    particle.pos += particle.vel * delta_time;
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
  uint index = gl_GlobalInvocationID.x;

  if (index >= data.length())
    return;

  update(data[index]);
}
)glsl"
