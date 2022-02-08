/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
 */

#pragma once

#include <iterator>

#include "Vector2.hpp"

namespace sm
{
template <sm::concepts::number T = double> class Rect
{
  public:
    // tag dispatching modes:
    static class default_init_t
    {
    } default_init;
    static class from_min_max_t
    {
    } from_min_max;
    static class centre_width_height_t
    {
    } centre_width_height;

    // iterator:
    struct Iterator
    {
        // using iterator_category = std::forward_iterator_tag;
        // using difference_type   = std::ptrdiff_t;
        // using value_type        = Vector2_t<T>;
        // using pointer           = Vector2_t<T>*;
        // using reference         = Vector2_t<T>&;

        constexpr Iterator(Vector2_t<T> indices, const Rect& rect)
            : m_indices{ indices }, m_rect{ rect }
        {
        }

        constexpr auto operator*() const { return m_indices; }

        constexpr Vector2_t<T>* operator->() { return &m_indices; }

        constexpr Vector2_t<T>& operator++()
        {
            if (m_indices.x == m_rect.m_max.x)
            {
                m_indices.x = m_rect.m_min.x;
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

        constexpr friend bool operator==(const Iterator& a, const Iterator& b)
        {
            return a.m_rect == b.m_rect and a.m_indices == b.m_indices;
        }

        constexpr friend bool operator!=(const Iterator& a, const Iterator& b)
        {
            return not(a == b);
        }

      private:
        Vector2_t<T> m_indices;
        const Rect& m_rect;
    };

    // constructors:
    constexpr Rect() noexcept : m_min{}, m_max{} {}

    constexpr Rect(Vector2_t<T> vec1, Vector2_t<T> vec2) noexcept
        : m_min{ std::min(vec1.x, vec2.x), std::min(vec1.y, vec2.y) }, m_max{
              std::max(vec1.x, vec2.x), std::max(vec1.y, vec2.y)
          }
    {
    }

    constexpr Rect(from_min_max_t, Vector2_t<T> min_, Vector2_t<T> max_) noexcept
        : m_min{ min_ }, m_max{ max_ }
    {
    }

    constexpr Rect(centre_width_height_t, Vector2_t<T> centre, T width, T height) noexcept
        : m_min{ centre - Vector2_t<T>{ -std::abs(width), -std::abs(height) } }, m_max{
              centre + Vector2_t<T>{ std::abs(width), std::abs(height) }
          }
    {
    }

    [[nodiscard]] constexpr auto min() const { return m_min; }
    [[nodiscard]] constexpr auto max() const { return m_max; }

    [[nodiscard]] constexpr auto operator[](size_t index) const
    {
        return m_min + convert_1d_to_2d(m_max - m_min + Vector2_t<T>{ 1, 1 }, index);
    }

    [[nodiscard]] constexpr auto begin() requires sm::concepts::integral<T>
    {
        return Iterator{ m_min, *this };
    }
    [[nodiscard]] constexpr auto end() requires sm::concepts::integral<T>
    {
        return Iterator{ Vector2_t<T>{ m_min.x, m_max.y + 1 }, *this };
    }
    [[nodiscard]] constexpr auto cbegin() const requires sm::concepts::integral<T>
    {
        return Iterator{ m_min, *this };
    }
    [[nodiscard]] constexpr auto cend() const requires sm::concepts::integral<T>
    {
        return Iterator{ Vector2_t<T>{ m_min.x, m_max.y + 1 }, *this };
    }

    [[nodiscard]] constexpr friend bool operator==(const Rect<T>& lhs, const Rect<T>& rhs)
    {
        return lhs.m_min == rhs.m_min and lhs.m_max == rhs.m_max;
    }

  private:
    // members:
    Vector2_t<T> m_min;
    Vector2_t<T> m_max;
};
} // namespace sm
