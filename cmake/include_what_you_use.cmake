# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

function(iwyu IWYU_TARGET)
    if(USE_IWYU)
        message(STATUS "samarium: nabled include-what-you-use")
        find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
        set_property(
            TARGET ${IWYU_TARGET} PROPERTY CXX_INCLUDE_WHAT_YOU_USE
                                           "${iwyu_path};-w;-Xiwyu;--verbose=2"
        )
    endif()
endfunction()
