/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "../graphics/Image.hpp"
#include "../util/format.hpp"

namespace sm::file
{
struct Text
{
};

struct Targa
{
};

auto read(Text, const std::filesystem::path& file_path) -> std::optional<std::string>;
auto read(const std::filesystem::path& file_path) -> std::optional<std::string>;

void export_to(Targa,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".tga");
} // namespace sm::file
