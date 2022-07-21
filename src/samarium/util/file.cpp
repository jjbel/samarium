/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <array>      // for to_array, array
#include <chrono>     // for filesystem
#include <filesystem> // for path
#include <fstream>    // for ifstream, ofstream, basic_ostream::write
#include <iterator>
#include <new> // for bad_alloc
#include <optional>
#include <string> // for string

#include "../core/types.hpp"     // for u8
#include "../graphics/Color.hpp" // for BGR_t, bgr
#include "../graphics/Image.hpp" // for Image
#include "../math/Vector2.hpp"   // for Dimensions

namespace sm::file
{
void export_tga(const Image& image, const std::filesystem::path& file_path)
{
    namespace fs = std::filesystem;

    const auto tga_header = std::to_array<u8>(
        {0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, static_cast<u8>(255 & image.dims.x),
         static_cast<u8>(255 & (image.dims.x >> 8)), static_cast<u8>(255 & image.dims.y),
         static_cast<u8>(255 & (image.dims.y >> 8)), 24, 32});

    const auto data = image.formatted_data(sm::bgr);

    std::ofstream(file_path)
        .write(reinterpret_cast<const char*>(&tga_header[0]), 18)
        .write(reinterpret_cast<const char*>(&data[0]),
               static_cast<std::streamsize>(data.size() * data[0].size()));
}

void export_to(const Image& image, const std::filesystem::path& file_path)
{
    const auto extension = file_path.extension();
    if (extension == ".tga") { export_tga(image, file_path); }
}

auto read(const std::filesystem::path& file_path) -> std::optional<std::string>
{
    if (std::filesystem::exists(file_path))
    {
        auto ifs = std::ifstream{file_path};
        return {std::string(std::istreambuf_iterator<char>{ifs}, {})};
    }
    else { return {}; }
}
} // namespace sm::file
