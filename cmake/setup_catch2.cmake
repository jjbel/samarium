# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

# From https://google.github.io/googletest/quickstart-cmake.html

function(setup_catch2)
    include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG f4af9f69265d009a457aa99d1075cfba78652a66 # v3.0.0-preview4
    )

    message(STATUS "Installed Catch2")

    FetchContent_MakeAvailable(Catch2)
endfunction(setup_catch2)
