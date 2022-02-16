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

    const auto tga_header = std::array<u8, 18>{0,
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
