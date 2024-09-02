# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/jjbel/samarium>

# Run the command ${RUN_AFTER_BUILD}
# after building ${RUN_AFTER_BUILD_DEPENDS}
# creates the dummy file running_${RUN_AFTER_BUILD_DEPENDS}.txt

# for example
# set RUN_AFTER_BUILD to "test/samarium_tests" and
# RUN_AFTER_BUILD_DEPENDS to samarium_tests
# to run samarium_tests after building it

if(RUN_AFTER_BUILD)
# TODO to implement
    message(WARNING "TO IMPLEMENT run after build: ${RUN_AFTER_BUILD}")

    # set(RUN_OUTPUT "${CMAKE_BINARY_DIR}/running_${RUN_AFTER_BUILD_DEPENDS}.txt")
    # add_custom_command(
    #     OUTPUT ${RUN_OUTPUT}
    #     COMMAND ${RUN_AFTER_BUILD}
    #     DEPENDS ${RUN_AFTER_BUILD_DEPENDS}
    #     USES_TERMINAL
    # )

    # # Create target which consume the command via DEPENDS.
    # add_custom_target(run ALL DEPENDS ${RUN_OUTPUT})

    # add_custom_target(run COMMAND ${RUN_AFTER_BUILD} USES_TERMINAL)
    # add_custom_command(TARGET ${RUN_AFTER_BUILD_DEPENDS} POST_BUILD COMMAND ${RUN_AFTER_BUILD} USES_TERMINAL)
endif()
