#pragma once

#include "samarium/core/types.hpp"

namespace sm::cuda
{
struct HostDevVec
{
    u64 count{};
    f32* host;
    f32* dev;

    u64 byte_size() const { return count * sizeof(float); }

    // void malloc_host() { host = (float*)malloc(byte_size()); }
    void malloc_dev() { cudaMalloc(&dev, byte_size()); }

    // void free_host() { free(host); }
    void free_dev() { cudaFree(dev); }

    void host2dev() { cudaMemcpy(dev, host, byte_size(), cudaMemcpyHostToDevice); }
    void dev2host() { cudaMemcpy(host, dev, byte_size(), cudaMemcpyDeviceToHost); }
};
} // namespace sm::cuda
