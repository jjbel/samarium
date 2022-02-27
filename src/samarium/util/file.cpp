/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <array>
#include <filesystem>
#include <fstream>

#include "samarium/graphics/Image.hpp"

#include "print.hpp"

namespace sm::file
{
void export_tga(const Image& image, const std::string& file_path)
{
    namespace fs = std::filesystem;

    const auto tga_header =
        std::array<u8, 18>{0,
                           0,
                           2,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           static_cast<u8>(255 & image.dims.x),
                           static_cast<u8>(255 & (image.dims.x >> 8)),
                           static_cast<u8>(255 & image.dims.y),
                           static_cast<u8>(255 & (image.dims.y >> 8)),
                           24,
                           32};

    const auto data = image.formatted_data(sm::bgr);

    std::ofstream{file_path}
        .write(reinterpret_cast<const char*>(&tga_header[0]), 18)
        .write(reinterpret_cast<const char*>(&data[0]),
               static_cast<std::streamsize>(data.size() * data[0].size()));
}

void export_to(const Image& image, const std::string& file_path)
{
    if (file_path.ends_with(".tga")) export_tga(image, file_path);
}
} // namespace sm::file
