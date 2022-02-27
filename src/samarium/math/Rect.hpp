/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <iterator>

#include "Vector2.hpp"

namespace sm
{
template <concepts::Number T = double_t> struct Rect
{
    Vector2_t<T> min;
    Vector2_t<T> max;


    // iterator:
    struct Iterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type        = Vector2_t<T>;
        using pointer           = Vector2_t<T>*;
        using reference         = Vector2_t<T>&;

        constexpr Iterator(Vector2_t<T> indices, const Rect& rect)
            : m_indices{indices}, m_rect{rect}
        {
        }

        constexpr auto operator*() const { return m_indices; }

        constexpr Vector2_t<T>* operator->() { return &m_indices; }

        constexpr Vector2_t<T>& operator++()
        {
            if (m_indices.x == m_rect.max.x)
            {
                m_indices.x = m_rect.min.x;
                m_indices.y++;
            }
            else
            {
                m_indices.x++;
            }
            return m_indices;
        }

        constexpr Vector2_t<T> operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr friend bool operator==(const Iterator& a,
                                         const Iterator& b) noexcept = default;

      private:
        Vector2_t<T> m_indices;
        const Rect& m_rect;
    };

    template <concepts::Number U> constexpr auto as() const
    {
        // very weird: https://stackoverflow.com/a/3505738/17100530
        return Rect<U>{min.template as<U>(), max.template as<U>()};
    }

    static constexpr auto find_min_max(Vector2_t<T> p1, Vector2_t<T> p2)
    {
        return Rect<T>{{math::min(p1.x, p2.x), math::min(p1.y, p2.y)},
                       {math::max(p1.x, p2.x), math::max(p1.y, p2.y)}};
    }

    static constexpr auto
    from_centre_width_height(Vector2_t<T> centre, T width, T height)
    {
        const auto vec = Vector2_t{.x = width, .y = height};
        return Rect{centre - vec, centre + vec};
    }

    [[nodiscard]] constexpr auto operator[](size_t index) const
    {
        return min + convert_1d_to_2d(max - min + Vector2_t<T>{1, 1}, index);
    }

    [[nodiscard]] constexpr auto begin() requires sm::concepts::Integral<T>
    {
        return Iterator{min, *this};
    }

    [[nodiscard]] constexpr auto end() requires sm::concepts::Integral<T>
    {
        return Iterator{Vector2_t<T>{min.x, max.y + 1}, *this};
    }

    [[nodiscard]] constexpr auto cbegin() const requires sm::concepts::Integral<T>
    {
        return Iterator{min, *this};
    }

    [[nodiscard]] constexpr auto cend() const requires sm::concepts::Integral<T>
    {
        return Iterator{Vector2_t<T>{min.x, max.y + 1}, *this};
    }

    [[nodiscard]] constexpr auto contains(const Vector2_t<T>& vec) const noexcept
    {
        return vec.x >= min.x && vec.x <= max.x && vec.y >= min.y &&
               vec.y <= max.y;
    }

    [[nodiscard]] constexpr friend bool
    operator==(const Rect<T>& lhs, const Rect<T>& rhs) noexcept = default;
};
} // namespace sm
