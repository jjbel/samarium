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
#include <execution>
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
        std::fill(std::execution::par_unseq, &this->data[0], &this->data[size_], init_value);
    }

    DynArray(const DynArray& arr) : m_size(arr.m_size), data(new T[arr.m_size]{})
    {
        std::copy(std::execution::par_unseq, arr.begin(), arr.end(), this->data);
    }

    DynArray(DynArray&& arr) noexcept : m_size(arr.m_size) { std::swap(data, arr.data); }

    // Operator overloads
    DynArray& operator=(const DynArray& arr)
    {
        std::copy(arr.begin(), arr.end(), this->data);
        return *this;
    }

    T& operator[](size_t index) noexcept { return this->data[index]; }

    const T& operator[](size_t index) const noexcept { return this->data[index]; }

    T& at(size_t index)
    {
        if (index >= this->m_size)
            throw std::out_of_range(
                fmt::format("sm::DynArray: index {} out of range for m_size {}", index,
                            this->m_size));
        return this->data[index];
    }

    const T& at(size_t index) const
    {
        if (index >= this->m_size)
            throw std::out_of_range(
                fmt::format("sm::DynArray: index {} out of range for m_size {}", index,
                            this->m_size));
        return this->data[index];
    }

    auto begin() noexcept { return static_cast<iterator>(&this->data[0]); }
    auto end() noexcept { return static_cast<iterator>(&this->data[this->m_size]); }

    auto begin() const noexcept { return static_cast<const_iterator>(&this->data[0]); }
    auto end() const noexcept
    {
        return static_cast<const_iterator>(&this->data[this->m_size]);
    }

    auto cbegin() const noexcept { return static_cast<const_iterator>(&this->data[0]); }
    auto cend() const noexcept
    {
        return static_cast<const_iterator>(&this->data[this->m_size]);
    }

    auto size() const noexcept { return this->m_size; }
    auto max_size() const noexcept { return this->m_size; } // for stl compatibility
    auto empty() const noexcept { return this->m_size == 0; }

    auto as_span() { return std::span{ this->begin(), this->m_size }; }

    auto fill(const T& value)
    {
        std::fill(std::execution::par_unseq, this->begin(), this->end(), value);
    }

    ~DynArray() { delete[] data; }

  private:
    const size_t m_size;
    T* data;
};
} // namespace sm
