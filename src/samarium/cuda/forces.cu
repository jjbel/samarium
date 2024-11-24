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

__global__ void add(u64 n, f32* x)
{
    for (int i = 0; i < n / 2; i++)
    {
        x[2 * i] += 1.0;
        x[2 * i + 1] += 3.0;
    }
}

void forces(const ForcesSettings& settings)
{
    auto watch = Stopwatch{};
    auto hdv   = HostDevVec{settings.count * 2, &settings.pos[0].x};
    hdv.malloc_dev();
    watch.print_reset("malloc_dev");

    hdv.host2dev();
    watch.print_reset("host2dev");

    add<<<1, 1>>>(hdv.count, hdv.dev);
    watch.print_reset("kernel");

    hdv.dev2host();
    watch.print_reset("dev2host");

    hdv.free_dev();
    watch.print_reset("free_dev");
}
}; // namespace sm::cuda
