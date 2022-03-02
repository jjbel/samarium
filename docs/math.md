# math

### In file: `samarium/math/math.hpp`

### Contents

- [Definition](#definition)
- [About](#about)
- [Example](#example)
- [Members](#members)

## Definition

```cpp
namespace sm::math{}
```

## About

Various mathematical functions forming the core of samarium

## Example

```cpp
using namespace sm::literals;
sm::print()
fmt::format("{} degrees is {} in radians", 36, 36_degrees);
```

## Members

### All members are marked `constexpr`, `inline`, `[[nodiscard]]` and `noexcept`

- `double sm::math::EPSILON = 1.e-4;`

- `template <sm::concepts::FloatingPoint T> bool sm::math::almost_equal(T a, T b);`

- `template <typename T> T sm::math::min(T value0, T value1);`

- `template <typename T> T sm::math::max(T value0, T value1);`

- `template <u32 n> auto sm::math::power(auto x);`

- `consteval double sm::literals::operator"" _degrees(long double angle);`

- `consteval double sm::literals::operator"" _radians(long double angle);`
