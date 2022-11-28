/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp"

namespace sm
{
// Credit (MIT Licensed):
// https://github.com/paweldac/source_location/blob/ff0002f92cdde3576ce02048dd9eb7823cabdc7b/include/source_location/source_location.hpp
struct SourceLocation
{
#ifdef __clang__
    static constexpr auto current(const char* fileName     = __builtin_FILE(),
                                  const char* functionName = __builtin_FUNCTION(),
                                  u32 lineNumber           = __builtin_LINE(),
                                  u32 columnOffset = __builtin_COLUMN()) noexcept -> SourceLocation
#elif defined(__GNUC__)
    static constexpr source_location current(const char* fileName     = __builtin_FILE(),
                                             const char* functionName = __builtin_FUNCTION(),
                                             const u32 lineNumber     = __builtin_LINE(),
                                             const u32 columnOffset   = 0) noexcept
#else
    // TODO MSVC?
    static constexpr source_location current(const char* fileName     = "unsupported",
                                             const char* functionName = "unsupported",
                                             u32 lineNumber           = 0,
                                             u32 columnOffset         = 0) noexcept
#endif
    {
        return SourceLocation(fileName, functionName, lineNumber, columnOffset);
    }

    SourceLocation(const SourceLocation&) = default;
    SourceLocation(SourceLocation&&)      = default;

    [[nodiscard]] constexpr auto file_name() const noexcept -> const char* { return fileName; }

    [[nodiscard]] constexpr auto function_name() const noexcept -> const char*
    {
        return functionName;
    }

    [[nodiscard]] constexpr auto line() const noexcept -> u32 { return lineNumber; }

    [[nodiscard]] constexpr auto column() const noexcept -> u32 { return columnOffset; }

  private:
    constexpr SourceLocation(const char* fileName,
                             const char* functionName,
                             u32 lineNumber,
                             u32 columnOffset) noexcept
        : fileName(fileName), functionName(functionName), lineNumber(lineNumber),
          columnOffset(columnOffset)
    {
    }

    const char* fileName;
    const char* functionName;
    const u32 lineNumber;
    const u32 columnOffset;
};
} // namespace sm
