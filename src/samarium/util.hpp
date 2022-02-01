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

#include <chrono>
#include <concepts>
#include <iterator>

#include "util.hpp"

namespace sm::util
{
template <typename T>
concept integral = std::is_integral_v<T>;

template <typename T>
concept floating_point = std::is_floating_point_v<T>;

template <typename T>
concept number = integral<T> || floating_point<T>;

template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template <class ContainerType>
concept Container_t = requires(ContainerType a, const ContainerType b)
{
    requires std::regular<ContainerType>;
    requires std::swappable<ContainerType>;
    requires std::destructible<typename ContainerType::value_type>;
    requires std::same_as<typename ContainerType::reference,
                          typename ContainerType::value_type&>;
    requires std::same_as<typename ContainerType::const_reference,
                          const typename ContainerType::value_type&>;
    requires std::forward_iterator<typename ContainerType::iterator>;
    requires std::forward_iterator<typename ContainerType::const_iterator>;
    requires std::signed_integral<typename ContainerType::difference_type>;
    requires std::same_as<
        typename ContainerType::difference_type,
        typename std::iterator_traits<typename ContainerType::iterator>::difference_type>;
    requires std::same_as<typename ContainerType::difference_type,
                          typename std::iterator_traits<
                              typename ContainerType::const_iterator>::difference_type>;
    {
        a.begin()
        } -> std::same_as<typename ContainerType::iterator>;
    {
        a.end()
        } -> std::same_as<typename ContainerType::iterator>;
    {
        b.begin()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        b.end()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.cbegin()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.cend()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.size()
        } -> std::same_as<typename ContainerType::size_type>;
    {
        a.max_size()
        } -> std::same_as<typename ContainerType::size_type>;
    {
        a.empty()
        } -> std::same_as<bool>;
};

size_t constexpr strlen(const char* str) { return *str ? 1u + strlen(str + 1) : 0u; }

// template <char ch> consteval uint8_t from_hex()
// {
//     if constexpr ('0' <= ch && ch <= '9') return ch - '0';
//     if constexpr ('a' <= ch && ch <= 'f') return ch - 'a' + 10;
//     if constexpr ('A' <= ch && ch <= 'F') return ch - 'a' + 10;
//     throw std::logic_error("hex character must be 0-9, a-f, or A-F");
// }
consteval uint8_t hex_to_int_safe(char ch)
{
    if ('0' <= ch && ch <= '9') return static_cast<uint8_t>(ch - '0');
    if ('a' <= ch && ch <= 'f') return static_cast<uint8_t>(ch - 'a' + 10);
    if ('A' <= ch && ch <= 'F') return static_cast<uint8_t>(ch - 'A' + 10);
    throw std::logic_error("hex character must be 0-9, a-f, or A-F");
}

uint8_t hex_to_int(char ch)
{
    if ('0' <= ch && ch <= '9') return static_cast<uint8_t>(ch - '0');
    if ('a' <= ch && ch <= 'f') return static_cast<uint8_t>(ch - 'a' + 10);
    if ('A' <= ch && ch <= 'F') return static_cast<uint8_t>(ch - 'A' + 10);
    throw std::logic_error("hex character must be 0-9, a-f, or A-F");
}

class Stopwatch
{
  public:
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    auto time() const
    {
        auto finish = std::chrono::steady_clock::now();
        return 1000 *
               std::chrono::duration_cast<std::chrono::duration<double>>(finish - start)
                   .count();
    }

    auto print() const { fmt::print("Took {:.3}ms\n", this->time()); }
};
} // namespace sm::util
