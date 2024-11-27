#pragma once

#include "samarium/core/types.hpp"

namespace sm::cuda
{
struct HostDevVec
{
    u64 count{};
    f32* host;
    f32* dev;

    u64 byte_size() const;

    // void malloc_host() { host = (float*)malloc(byte_size()); }
    void malloc_dev();

    // void free_host() { free(host); }
    void free_dev();

    void host2dev();
    void dev2host();
};
} // namespace sm::cuda
