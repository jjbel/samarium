/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <array>      // for to_array, array
#include <filesystem> // for path
#include <fstream>    // for ifstream, ofstream, basic_ostream::write
#include <iterator>   // for ifstreambuf_iterator
#include <optional>   // for optional
#include <string>     // for string

#include "fpng/fpng.hpp"
#include "stb_image_write.h"

#include "../core/types.hpp"     // for u8
#include "../graphics/Color.hpp" // for BGR_t, bgr
#include "../graphics/Image.hpp" // for Image
#include "../math/Vector2.hpp"   // for Dimensions

#include "file.hpp"

namespace sm::file
{
auto read(Text, const std::filesystem::path& file_path) -> std::optional<std::string>
{
    if (std::filesystem::exists(file_path))
    {
        auto ifs = std::ifstream{file_path};
        return {std::string(std::istreambuf_iterator<char>{ifs}, {})};
    }
    else { return {}; }
}

auto read(const std::filesystem::path& file_path) -> std::optional<std::string>
{
    return read(Text{}, file_path);
}

void export_to(Targa, const Image& image, const std::filesystem::path& file_path)
{
    const auto tga_header = std::to_array<u8>(
        {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<u8>(255 & image.dims.x),
         static_cast<u8>(255 & (image.dims.x >> 8)), static_cast<u8>(255 & image.dims.y),
         static_cast<u8>(255 & (image.dims.y >> 8)), 24, 32});

    const auto data = image.formatted_data(sm::bgr);

    std::ofstream(file_path, std::ios::binary)
        .write(reinterpret_cast<const char*>(&tga_header[0]), 18)
        .write(reinterpret_cast<const char*>(&data[0]),
               static_cast<std::streamsize>(data.size() * data[0].size()));
}

void export_to(Bmp, const Image& image, const std::filesystem::path& file_path)
{
    stbi_write_bmp(file_path.string().c_str(), static_cast<i32>(image.dims.x),
                   static_cast<i32>(image.dims.y), 4 /* RGBA */,
                   static_cast<const void*>(&image.front()));
}
} // namespace sm::file
