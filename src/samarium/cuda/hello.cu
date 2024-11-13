#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include <cmath>
#include <iostream>

#include "hello.hpp"

#include "samarium/util/Stopwatch.hpp"

#define FMT_UNICODE 0
#include "fmt/format.h"

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
    std::cout << watch.seconds() << '\n';

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    for (int i = 0; i < 100; i++) { std::cout << y[i] << ' '; }
    std::cout << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    // return 0;
}

void thrust_benchmark_1(u64 count)
{
    auto watch = sm::Stopwatch{};

    const auto print_time = [&](const std::string& str)
    {
        fmt::print("{:18}: {}\n", str, watch.str_ms());
        watch.reset();
    };

    std::cout << "count: " << count << std::endl;
    thrust::default_random_engine rng(1337);
    thrust::uniform_real_distribution<float> dist;
    print_time("make rng");

    thrust::host_vector<float> h_vec(count);
    print_time("make host vec");

    thrust::host_vector<float> d_vec(count);
    print_time("make device vec");

    thrust::generate(h_vec.begin(), h_vec.end(), [&] { return float(dist(rng)); });
    print_time("random");

    // Transfer data to the device.
    d_vec = h_vec;
    print_time("host > device");

    // Sort data on the device.
    thrust::sort(d_vec.begin(), d_vec.end());
    print_time("sort");
    // Transfer data back to host.
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    print_time("device > host");
    std::cout << std::endl;
}
} // namespace sm
