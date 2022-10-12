/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <filesystem>       // for path
#include <initializer_list> // for initializer_list
#include <string>           // for string, operator+

#include "samarium/graphics/Image.hpp" // for Image
#include "samarium/math/Vector2.hpp"   // for Dimensions
#include "samarium/util/format.hpp"    // for date_time_str

#include "Expected.hpp"  // for Expected
#include "fpng/fpng.hpp" // for fpng_encode_image_to_file

namespace sm::file
{
struct Text
{
};

struct Targa
{
};

struct Pam
{
};

struct Png
{
};

struct Bmp
{
};

template <typename T> using ExpectedFile = Expected<T, std::string>;

auto read(Text, const std::filesystem::path& file_path) -> ExpectedFile<std::string>;

auto read(const std::filesystem::path& file_path) -> ExpectedFile<std::string>;

auto read_image(const std::filesystem::path& file_path) -> ExpectedFile<Image>;


void write(Targa,
           const Image& image,
           const std::filesystem::path& file_path = date_time_str() + ".tga");

/**
 * @brief               Write image to file_path in the NetBPM PAM format
 *
 * @param  image
 * @param  file_path
 * @details See https://en.wikipedia.org/wiki/Netpbm#PAM_graphics_format
 */
void write(Pam,
           const Image& image,
           const std::filesystem::path& file_path = date_time_str() + ".pam");

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
    -> Expected<std::filesystem::path, std::string>;

auto find(const std::string& file_name, std::span<std::filesystem::path> search_paths)
    -> Expected<std::filesystem::path, std::string>;

auto find(const std::string& file_name, std::initializer_list<std::filesystem::path> search_paths)
    -> Expected<std::filesystem::path, std::string>;
} // namespace sm::file


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_FILE_IMPL)

#include <array>      // for to_array, array
#include <cstring>    // for memcpy
#include <filesystem> // for path
#include <fstream>    // for ifstream, ofstream, basic_ostream::write
#include <iterator>   // for ifstreambuf_iterator
#include <string>     // for string

#include "fmt/os.h"
#include "range/v3/algorithm/copy.hpp"
#include "samarium/util/Stopwatch.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#include "samarium/core/inline.hpp"    // for SM_INLINE
#include "samarium/core/types.hpp"     // for u8
#include "samarium/graphics/Color.hpp" // for BGR_t, bgr
#include "samarium/graphics/Image.hpp" // for Image
#include "samarium/math/Extents.hpp"   // for range
#include "samarium/math/Vector2.hpp"   // for Dimensions

#include "fpng/fpng.hpp"

namespace sm::file
{
SM_INLINE auto read(Text, const std::filesystem::path& file_path) -> ExpectedFile<std::string>
{
    if (!std::filesystem::exists(file_path))
    {
        return tl::make_unexpected(fmt::format("{} does not exist", file_path));
    }
    else if (!std::filesystem::is_regular_file(file_path))
    {
        return tl::make_unexpected(fmt::format("{} is not a file", file_path));
    }
    else
    {
        auto ifs = std::ifstream{file_path};
        return {std::string(std::istreambuf_iterator<char>{ifs}, {})};
    }
}

SM_INLINE auto read(const std::filesystem::path& file_path) -> ExpectedFile<std::string>
{
    return read(Text{}, file_path);
}

SM_INLINE auto read_image(const std::filesystem::path& file_path) -> ExpectedFile<Image>
{
    if (!std::filesystem::exists(file_path))
    {
        return tl::make_unexpected(fmt::format("{} does not exist", file_path));
    }
    if (!std::filesystem::is_regular_file(file_path))
    {
        return tl::make_unexpected(fmt::format("{} is not a file", file_path));
    }

    auto width         = 0;
    auto height        = 0;
    auto channel_count = 0;

    const auto data = stbi_load(file_path.string().c_str(), &width, &height, &channel_count, 0);

    if (!data) { return tl::make_unexpected(fmt::format("Error while reading {}", file_path)); }

    auto image = Image{{static_cast<u64>(width), static_cast<u64>(height)}};

    if (channel_count == 4) { std::memcpy(&image.front(), data, image.size() * 4); }
    else if (channel_count == 3)
    {
        for (auto i : range(image.size()))
        {
            image[i].r = data[i * 3];
            image[i].g = data[i * 3 + 1];
            image[i].b = data[i * 3 + 2];
        }
    }
    else if (channel_count == 2)
    {
        for (auto i : range(image.size()))
        {
            const auto value = data[i * 2];
            image[i].r       = value;
            image[i].g       = value;
            image[i].b       = value;
            image[i].a       = data[i * 2 + 1];
        }
    }
    else if (channel_count == 1)
    {
        for (auto i : range(image.size()))
        {
            const auto value = data[i];
            image[i].r       = value;
            image[i].g       = value;
            image[i].b       = value;
        }
    }
    else
    {
        return tl::make_unexpected(
            fmt::format("{} has unknown channel count: {}", file_path, channel_count));
    }

    return {image};
}

SM_INLINE void write(Targa, const Image& image, const std::filesystem::path& file_path)
{
    const auto header = std::to_array<u8>(
        {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<u8>(255 & image.dims.x),
         static_cast<u8>(255 & (image.dims.x >> 8)), static_cast<u8>(255 & image.dims.y),
         static_cast<u8>(255 & (image.dims.y >> 8)), 24, 32});

    const auto data = image.formatted_data(bgr);

    std::ofstream(file_path, std::ios::binary)
        .write(reinterpret_cast<const char*>(&header[0]), header.size())
        .write(reinterpret_cast<const char*>(&data[0]),
               static_cast<std::streamsize>(data.size() * data[0].size()));
}

SM_INLINE void write(Pam, const Image& image, const std::filesystem::path& file_path)
{
    const auto header = fmt::format(R"(P7
WIDTH {}
HEIGHT {}
DEPTH 4
MAXVAL 255
TUPLTYPE RGB_ALPHA
ENDHDR
)",
                                    image.dims.x, image.dims.y);

    std::ofstream(file_path, std::ios::binary)
        .write(&header[0], static_cast<std::streamsize>(header.size()))
        .write(reinterpret_cast<const char*>(&image[0]),
               static_cast<std::streamsize>(image.byte_size()));
}

SM_INLINE void write(Bmp, const Image& image, const std::filesystem::path& file_path)
{
    stbi_write_bmp(file_path.string().c_str(), static_cast<i32>(image.dims.x),
                   static_cast<i32>(image.dims.y), 4 /* RGBA */,
                   static_cast<const void*>(&image.front()));
}


SM_INLINE auto find(const std::string& file_name, const std::filesystem::path& directory)
    -> Expected<std::filesystem::path, std::string>
{
    for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(directory))
    {
        if (dir_entry.is_regular_file() && dir_entry.path().filename() == file_name)
        {
            return {std::filesystem::canonical(dir_entry)};
        }
    }

    return tl::make_unexpected(fmt::format("File not found: {}", file_name));
}

SM_INLINE auto find(const std::string& file_name, std::span<std::filesystem::path> search_paths)
    -> Expected<std::filesystem::path, std::string>
{
    for (const auto& path : search_paths)
    {
        if (std::filesystem::is_directory(path))
        {
            if (const auto found_path = find(file_name, path)) { return {found_path}; }
        }
        else if (std::filesystem::is_regular_file(path))
        {
            if (std::filesystem::exists(path)) { return {std::filesystem::canonical(path)}; }
        }
    }

    return tl::make_unexpected(fmt::format("File not found: {}", file_name));
}

SM_INLINE auto find(const std::string& file_name,
                    std::initializer_list<std::filesystem::path> search_paths)
    -> Expected<std::filesystem::path, std::string>
{
    for (const auto& path : search_paths)
    {
        if (std::filesystem::is_directory(path))
        {
            if (const auto found_path = find(file_name, path)) { return {found_path}; }
        }
        else if (std::filesystem::is_regular_file(path))
        {
            if (std::filesystem::exists(path)) { return {std::filesystem::canonical(path)}; }
        }
    }

    return tl::make_unexpected(fmt::format("File not found: {}", file_name));
}
} // namespace sm::file

#endif