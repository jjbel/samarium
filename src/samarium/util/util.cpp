/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "util.hpp"
#include "date.hpp"

namespace sm::util
{
std::string get_date_filename(const std::string& extension)
{
    return date::format("%F_%T.", std::chrono::system_clock::now()) + extension;
}
} // namespace sm::util
