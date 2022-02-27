# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.16)

function(use_valgrind test_name test_program_path)
    find_program (VALGRIND_PATH valgrind)

    if (VALGRIND_PATH)
        message(STATUS "valgrind found: ${VALGRIND_PATH}")
        add_test(NAME ${test_name} COMMAND valgrind --leak-check=yes "${test_program_path}")
    else()
        message(STATUS "valgrind, find not found")
    endif()
endfunction()
