# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/conan.lock)
    find_program(CONAN_EXE conan REQUIRED)

    message(STATUS "Installing Conan dependencies... (this may take a few minutes)")
    execute_process(
        COMMAND ${CONAN_EXE} install . -if ${CMAKE_CURRENT_BINARY_DIR} -b missing
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_QUIET
    )
    set(CONAN_CMAKE_SILENT_OUTPUT True)
else()
    message(STATUS "Conan dependencies already installed")
endif()
