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

template <typename T> using ExpectedFile = tl::expected<T, std::string>;

auto read(Text, const std::filesystem::path& file_path) -> ExpectedFile<std::string>;

auto read(const std::filesystem::path& file_path) -> ExpectedFile<std::string>;

auto read_image(const std::filesystem::path& file_path) -> ExpectedFile<Image>;


void write(Targa,
           const Image& image,
           const std::filesystem::path& file_path = date_time_str() + ".tga");

inline void
write(Png, const Image& image, const std::filesystem::path& file_path = date_time_str() + ".png")
{
    fpng::fpng_encode_image_to_file(
        file_path.string().c_str(), static_cast<const void*>(&image.front()),
        static_cast<u32>(image.dims.x), static_cast<u32>(image.dims.y), 4U);
}

void write(Bmp,
           const Image& image,
           const std::filesystem::path& file_path = date_time_str() + ".bmp");

auto find(const std::string& file_name,
          const std::filesystem::path& directory = std::filesystem::current_path())
    -> tl::expected<std::filesystem::path, std::string>;

auto find(const std::string& file_name, std::span<std::filesystem::path> search_paths)
    -> tl::expected<std::filesystem::path, std::string>;

auto find(const std::string& file_name, std::initializer_list<std::filesystem::path> search_paths)
    -> tl::expected<std::filesystem::path, std::string>;
} // namespace sm::file
