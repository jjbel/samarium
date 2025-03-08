#include "HashGrid.hpp"

namespace sm::cuda
{
u64 HashGrid::get_index(Indices v)
{
    return v.x + (v.y + height + 1) * width;
}
} // namespace sm::cuda
