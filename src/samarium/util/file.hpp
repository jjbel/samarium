/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <filesystem>
#include <string>

#include "fpng/fpng.hpp"
#include "tl/expected.hpp"

#include "samarium/graphics/Image.hpp"
#include "samarium/util/format.hpp"

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

struct FileNotFoundError
{
};

auto read(Text, const std::filesystem::path& file_path)
    -> tl::expected<std::string, FileNotFoundError>;
auto read(const std::filesystem::path& file_path) -> tl::expected<std::string, FileNotFoundError>;

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

auto find(const std::string& file_name,
          const std::filesystem::path& directory = std::filesystem::current_path())
    -> tl::expected<std::filesystem::path, FileNotFoundError>;

auto find(const std::string& file_name, std::span<std::filesystem::path> search_paths)
    -> tl::expected<std::filesystem::path, FileNotFoundError>;

auto find(const std::string& file_name, std::initializer_list<std::filesystem::path> search_paths)
    -> tl::expected<std::filesystem::path, FileNotFoundError>;
} // namespace sm::file
