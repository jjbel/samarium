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

#include <algorithm>
#include <iterator>
#include <span>

namespace sm
{
template <typename T> class DynArray
{
  public:
    // Container types
    using value_type      = T;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = T const*;
    using difference_type = std::ptrdiff_t;
    using size_type       = std::size_t;

    // Constructors
    DynArray(size_t size_ = {}) : m_size(size_), data(new T[size_]{}) {}

    DynArray(size_t size_, T init_value) : m_size(size_), data(new T[size_]{})
    {
        std::fill(&this->data[0], &this->data[size_], init_value);
    }

    DynArray(const DynArray& field) : m_size(field.m_size), data(new T[field.m_size]{})
    {
        std::copy(field.begin(), field.end(), this->data);
    }

    DynArray(DynArray&& field) noexcept : m_size(field.m_size)
    {
        std::swap(data, field.data);
    }

    // Operator overloads
    DynArray& operator=(const DynArray& field)
    {
        std::copy(field.begin(), field.end(), this->data);
        return *this;
    }

    T& operator[](size_t index)
    {
        if (index >= this->m_size)
            throw std::out_of_range(
                fmt::format("sm::DynArray: index {} out of range for m_size {}", index,
                            this->m_size));
        return this->data[index];
    }

    const T& operator[](size_t index) const
    {
        if (index >= this->m_size)
            throw std::out_of_range(
                fmt::format("sm::DynArray: index {} out of range for m_size {}", index,
                            this->m_size));
        return this->data[index];
    }

    auto begin() { return iterator(&this->data[0]); }
    auto end() { return iterator(&this->data[this->m_size]); }

    auto begin() const { return const_iterator(&this->data[0]); }
    auto end() const { return const_iterator(&this->data[this->m_size]); }

    auto cbegin() const { return const_iterator(&this->data[0]); }
    auto cend() const { return const_iterator(&this->data[this->m_size]); }

    auto size() const { return this->m_size; }
    auto max_size() const { return this->m_size; } // for stl compatibility
    auto empty() const { return this->m_size == 0; }

    auto as_span() { return std::span{ this->begin(), this->m_size }; }

    ~DynArray()
    {
        delete[] data;
        fmt::print("Called DynArray destructor\n");
    }

  private:
    T* data;
    const size_t m_size;
};

// template <typename T>
// inline bool operator==(const DynArray<T>& lhs, const DynArray<T>& rhs)
// {
//     return lhs.m_size == rhs.m_size && std::equal(lhs.cbegin(), lhs.cend(),
//     rhs.cbegin());
// }

// template <typename T>
// inline bool operator!=(const DynArray<T>& lhs, const DynArray<T>& rhs)
// {
//     return !operator==(lhs, rhs);
// }
} // namespace sm
