# random

### In file: `samarium/util/random.hpp`

### Contents

- [Definition](#definition)
- [About](#about)
- [Example](#example)
- [Members](#members)

## Definition

```cpp
namespace sm::random{}
```

## About

Generate random numbers, objects...

`sm::random` works by precomputing random `double`s and storing them in a cache (`sm::random::cache`)

This is much faster than creating random numbers in a hot loop.

To resize this cache, call `sm::random::resize_cache(/* size_t */ size)`

## Example

```cpp
for(auto i : std::views::iota(1,10)
{
    sm::print(sm::random(), sm::rand_vector());
}
```

## Members

### All reside in namespace `sm::random`

- `static size_t cache_length`: length of cache

- `static size_t current`: Current index in cache

- `static std::vector<double> cache`

- `void fill_cache(size_t size)`

- `double random()`

- `template <typename T> T rand_range(Extents<T> range)`

- `Vector2 rand_vector(Rect<double> bounding_box)`

- `Vector2 rand_vector(Extents<f64> radius_range, Extents<f64> angle_range)`
