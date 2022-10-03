# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project
# homepage:<https://github.com/strangeQuark1041/samarium>

if(BUILD_DOCS)
    find_program(PYTHON_EXE NAMES python python3 REQUIRED)

    execute_process(
        COMMAND ${PYTHON_EXE} -m pip install -r src/requirements.txt --quiet
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        OUTPUT_QUIET
    )

    execute_process(
        COMMAND ${PYTHON_EXE} src/make.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
        OUTPUT_QUIET
    )

    message(STATUS "Built docs, view 'docs/build/html/index.html'")
endif()

if(BUILD_DOCS_TARGET)
    find_program(PYTHON_EXE NAMES python python3 REQUIRED)

    add_custom_target(
        docs
        COMMAND ${PYTHON_EXE} -m pip install -r src/requirements.txt --quiet
        COMMAND ${PYTHON_EXE} src/make.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
    )

    message(
        STATUS
            "Build the docs target, view 'docs/src/build/html/index.html' after building"
    )
endif()
