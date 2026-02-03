#include <cmath>

#define FMT_UNICODE 0
#include "fmt/format.h"
#include "samarium/util/Stopwatch.hpp"
#include "cuda.hpp"

namespace sm
{
// Kernel function to add the elements of two arrays
__global__ void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}

void test()
{
    auto watch = sm::Stopwatch{};

    int N = 1'000'000;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    watch.reset();
    add<<<1, 256>>>(N, x, y); // Run kernel on 1M elements on the GPU
    cudaDeviceSynchronize();  // Wait for GPU to finish before accessing on host
    watch.print();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i] - 3.0f));
    fmt::print("Max error: {}\n", maxError);
    for (int i = 0; i < 100; i++) { fmt::print("{} ", y[i]); }
    fmt::print("\n");

    // Free memory
    cudaFree(x);
    cudaFree(y);
}
} // namespace sm
