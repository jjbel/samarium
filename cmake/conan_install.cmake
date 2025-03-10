# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/jjbel/samarium>

# install dependencies by running 'conan install'

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/conan_toolchain.cmake AND RUN_CONAN)
    find_program(CONAN_EXE conan REQUIRED)

    set(DEPS_OPTION "")
    if(BUILD_UNIT_TESTS OR BUILD_BENCHMARKS)
        set(DEPS_OPTION "-o samarium/*:build_tests=True")
    endif()

    message(STATUS "samarium: installing dependencies ... (this may take a few seconds)")
    execute_process(
        # TODO conan args: make sure they're correct
        COMMAND ${CONAN_EXE} install .
        -b missing
        -vquiet
        # TODO how come not using -of ?
        # -if ${CMAKE_CURRENT_BINARY_DIR}
        # -pr:b=default -pr=default
        ${DEPS_OPTION}
        -s build_type=${CMAKE_BUILD_TYPE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        # OUTPUT_QUIET
        # TODO output_quiet not working? using -vquiet
        # TODO how to silence output from individual lib files
    )

    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
else()
    message(STATUS "samarium: dependencies already installed")
endif()
