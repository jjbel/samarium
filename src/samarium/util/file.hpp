/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../graphics/Image.hpp"
#include "../util/print.hpp"
#include "../util/util.hpp"

namespace sm::file
{
void export_tga(const Image& image, const std::string& file_path = util::get_date_filename("tga"));

void export_to(const Image& image, const std::string& file_path = util::get_date_filename("tga"));
} // namespace sm::file
