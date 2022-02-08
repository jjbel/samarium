# `sm::math::Vector2_t`

```cpp
template <sm::concepts::number T> class sm::Vector2_t;
using sm::Vector2 = Vector2_t<double>;
```

## About

A `Vector2_t` holds 2 values - an x and a y coordinate. It represents a Euclidean vector, or arrow in space.

A `Vector2` is a vector of doubles. As it is used frequently, it is a typedef for `Vector2_t<T>`

## Example

```cpp
auto vec = Vector2{2, 3};
fmt::print("Vector = {}", vec); // Vector = [2, 3]
```

## Members

1. `T x` : x-coordinate

2. `T y` : y-coordinate

## Constructors

1. `constexpr Vector2_t() noexcept` : Default constructor. Initializes `x` and `y` to 0

1. `constexpr Vector2_t(T x_, T y_) noexcept` : Construct from an x and a y coordinate

## Literals

1. `consteval auto operator"" _x(long double x);` : constructs `Vector2{x, 0}`

2. `consteval auto operator"" _y(long double y);` : constructs `Vector2{0, y}`

## Member Functions

1. `constexpr double length() const noexcept;` : length of the vector

2. `constexpr double length_sq() const noexcept;` : length squared of the vector. Avoids the `sqrt` to save computing time

3. `constexpr double angle() const noexcept;` : angle of the vector from the positive x-axis

## Operators

`Vector2_t` is overloaded on common mathematical operators for both other vectors and scalars (`double`s). Operations are conducted element wise. **This means that multiplication is element wise,** and not the dot or cross product.

Given 2 `Vector2_t<T>`s, `lhs` and `rhs`

1. `constexpr Vector2_t<T> operator+=(rhs) noexcept`

2. `constexpr Vector2_t<T> operator-=(rhs) noexcept`

3. `constexpr Vector2_t<T> operator*=(rhs) noexcept`

4. `constexpr Vector2_t<T> operator*=(T num) noexcept`

5. `constexpr Vector2_t<T> operator/=(rhs) noexcept`

6. `constexpr Vector2_t<T> operator/=(T num) noexcept`

7. `[[nodiscard]] constexpr inline bool operator==(lhs, rhs) noexcept;`

8. `[[nodiscard]] constexpr inline bool operator!=(lhs, rhs);`

9. `[[nodiscard]] constexpr inline Vector2_t<T> operator+(lhs, rhs) noexcept;`

10. `[[nodiscard]] constexpr inline Vector2_t<T> operator-(lhs, rhs) noexcept;`

11. `[[nodiscard]] constexpr inline Vector2_t<T> operator*(lhs, rhs) noexcept;`

12. `[[nodiscard]] constexpr inline Vector2_t<T> operator*(lhs, T num) noexcept;`

13. `[[nodiscard]] constexpr inline Vector2_t<T> operator*(T num, rhs) noexcept;`

14. `[[nodiscard]] constexpr inline Vector2_t<T> operator/(lhs, rhs) noexcept;`

15. `[[nodiscard]] constexpr inline Vector2_t<T> operator/(lhs, T num) noexcept;`
