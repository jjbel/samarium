#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#define FMT_UNICODE 0
#include "fmt/format.h"

#include "samarium/util/Stopwatch.hpp"

#include "cuda.hpp"

namespace sm::cuda
{
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
} // namespace sm::cuda