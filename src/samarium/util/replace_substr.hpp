/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include <string>
#include <string_view>

namespace sm::util
{
auto replace_substr_inplace(std::string& source, std::string_view old_str, std::string_view new_str)
{
    source.replace(source.find(old_str), old_str.size(), new_str);
}

auto replace_substr(std::string_view source, std::string_view old_str, std::string_view new_str)
{
    auto copy = std::string(source);
    copy.replace(copy.find(old_str), old_str.size(), new_str);
    return copy;
}
} // namespace sm::util
