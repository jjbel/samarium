#pragma once

#include "HostDevVec.hpp"

namespace sm::cuda
{
u64 HostDevVec::byte_size() const { return count * sizeof(float); }

// void HostDevVec::malloc_host() { host = (float*)malloc(byte_size()); }
void HostDevVec::malloc_dev() { cudaMalloc(&dev, byte_size()); }

// void HostDevVec::free_host() { free(host); }
void HostDevVec::free_dev() { cudaFree(dev); }

void HostDevVec::host2dev() { cudaMemcpy(dev, host, byte_size(), cudaMemcpyHostToDevice); }
void HostDevVec::dev2host() { cudaMemcpy(host, dev, byte_size(), cudaMemcpyDeviceToHost); }
} // namespace sm::cuda
