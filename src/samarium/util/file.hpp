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

#include "fpng/fpng.hpp"

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

struct Png
{
};

struct Bmp
{
};

namespace
{
static const auto inited = []()
{
    fpng::fpng_init();
    return true;
}();
}

auto read(Text, const std::filesystem::path& file_path) -> std::optional<std::string>;
auto read(const std::filesystem::path& file_path) -> std::optional<std::string>;

void export_to(Targa,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".tga");

void export_to(Png,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".png");

void export_to(Bmp,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".bmp");
} // namespace sm::file
