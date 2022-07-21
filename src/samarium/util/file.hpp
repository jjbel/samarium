/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <filesystem>
#include <string>

#include "../graphics/Image.hpp"
#include "../util/util.hpp"

namespace sm::file
{
void export_tga(const Image& image,
                const std::filesystem::path& file_path = date_time_str() + ".tga");

void export_to(const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".tga");

auto read(const std::filesystem::path& file_path) -> std::optional<std::string>;
} // namespace sm::file
