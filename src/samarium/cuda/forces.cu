#include "cuda.hpp"

#include "HostDevVec.cuh"

#include "samarium/util/Stopwatch.hpp"

namespace sm::cuda
{
void forces_host(const ForcesSettings& settings)
{
    float* pos_ptr = &settings.pos[0].x;
    float* acc_ptr = &settings.acc[0].x;

    for (u64 i = 0; i < settings.count; i++)
    {
        pos_ptr[2 * i] += 1.0;
        pos_ptr[2 * i + 1] += 2.0;
    }
}

__global__ void forces_kernel(u64 n, f32* pos, f32* acc, f32 strength, f32 max_force)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) { return; }

    const auto this_x = pos[2 * i];
    const auto this_y = pos[2 * i + 1];

    auto force_x = f32{};
    auto force_y = f32{};

    // https://docs.nvidia.com/cuda/cuda-math-api/index.html

    for (u64 j = 0; j < i; j++)
    {
        const auto dx       = pos[2 * j] - this_x;
        const auto dy       = pos[2 * j + 1] - this_y;
        const auto one_by_r = rhypotf(dx, dy);
        const auto force    = fminf(strength * one_by_r * one_by_r * one_by_r, max_force);
        force_x += dx * force;
        force_y += dy * force;
    }
    for (u64 j = i + 1; j < n; j++)
    {
        const auto dx       = pos[2 * j] - this_x;
        const auto dy       = pos[2 * j + 1] - this_y;
        const auto one_by_r = rhypotf(dx, dy);
        const auto force    = fminf(strength * one_by_r * one_by_r * one_by_r, max_force);
        force_x += dx * force;
        force_y += dy * force;
    }

    acc[2 * i]     = force_x;
    acc[2 * i + 1] = force_y;
    // x[2 * i + 1] += 3.0;
}


void forces(const ForcesSettings& settings)
{
    auto watch = Stopwatch{};

    auto pos = HostDevVec{settings.count * 2, &settings.pos[0].x};
    auto acc = HostDevVec{settings.count * 2, &settings.acc[0].x};
    if (!pos.dev) { pos.malloc_dev(); }
    if (!acc.dev) { acc.malloc_dev(); }
    watch.print_reset("\nmalloc_dev");

    pos.host2dev();
    watch.print_reset("host2dev");

    // 1 for each Vec2
    forces_kernel<<<(settings.count + 255) / 256, 256>>>(settings.count, pos.dev, acc.dev,
                                                         settings.strength, settings.max_force);
    watch.print_reset("kernel");
    // TODO not accurate: kernel launch is async https://stackoverflow.com/a/12793124

    pos.dev2host();
    acc.dev2host();
    watch.print_reset("dev2host");

    if (!pos.dev) { pos.free_dev(); }
    if (!acc.dev) { acc.free_dev(); }
    watch.print_reset("free_dev");
}
}; // namespace sm::cuda
