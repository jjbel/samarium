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


// namespace
// {
// auto png_init()
// {
//     fpng::fpng_init();
//     return true;
// }
// static const bool b = png_init();
// } // namespace


auto read(Text, const std::filesystem::path& file_path) -> std::optional<std::string>;
auto read(const std::filesystem::path& file_path) -> std::optional<std::string>;

void export_to(Targa,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".tga");

inline void export_to(Png,
                      const Image& image,
                      const std::filesystem::path& file_path = date_time_str() + ".png")
{
    fpng::fpng_encode_image_to_file(
        file_path.string().c_str(), static_cast<const void*>(&image.front()),
        static_cast<u32>(image.dims.x), static_cast<u32>(image.dims.y), 4U);
}

void export_to(Bmp,
               const Image& image,
               const std::filesystem::path& file_path = date_time_str() + ".bmp");
} // namespace sm::file
