/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
 */

#pragma once

#include <array>
#include <chrono>
#include <date/date.h>
#include <filesystem>
#include <fstream>

#include "Image.hpp"
#include "print.hpp"

namespace sm::file
{
auto write(const Image& image,
           std::string file_path      = "",
           const bool shouldOverwrite = false)
{
    namespace fs = std::filesystem;

    file_path = file_path.length() == 0
                    ? date::format("%F_%T.tga", std::chrono::system_clock::now())
                    : file_path;

    if (fs::exists(fs::path(file_path)) and !shouldOverwrite)
    {
        util::error('\'', file_path,
                    "' already exists.\nPass `true` as last argument of "
                    "sm::util::write() to overwrite\n");
        return false;
    }

    const std::array<unsigned char, 18> tga_header = {
        0,
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
        static_cast<unsigned char>(255 & image.dims.x),
        static_cast<unsigned char>(255 & (image.dims.x >> 8)),
        static_cast<unsigned char>(255 & image.dims.y),
        static_cast<unsigned char>(255 & (image.dims.y >> 8)),
        24,
        32
    };

    // const auto [pixels, length] = image.formatted_data(sm::BGR);
    const auto data = image.formatted_data(sm::BGR);

    if (std::ofstream out_file{ file_path })
    {
        out_file.write(reinterpret_cast<const char*>(&tga_header[0]), 18);
        out_file.write(reinterpret_cast<const char*>(&data[0]),
                       static_cast<std::streamsize>(data.size()));
        fmt::print("Size: {};\n", static_cast<std::streamsize>(data.size()));
        return true;
    }
    else
    {
        fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold,
                   "Error: unable to open \"{}\"\n", file_path);
    }

    return false;
}
} // namespace sm::file
