# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

# Run the command ${RUN_AFTER_BUILD}
# after building ${RUN_AFTER_BUILD_DEPENDS}
# creates the dummy file running_${RUN_AFTER_BUILD_DEPENDS}.txt

# for example
# set RUN_AFTER_BUILD to "test/samarium_tests" and
# RUN_AFTER_BUILD_DEPENDS to samarium_tests
# to run samarium_tests after building it

if(RUN_AFTER_BUILD)
    set(RUN_OUTPUT "${CMAKE_BINARY_DIR}/running_${RUN_AFTER_BUILD_DEPENDS}.txt")
    add_custom_command(
        OUTPUT ${RUN_OUTPUT}
        COMMAND ${RUN_AFTER_BUILD}
        DEPENDS ${RUN_AFTER_BUILD_DEPENDS}
    )

    # Create target which consume the command via DEPENDS.
    add_custom_target(run ALL DEPENDS ${RUN_OUTPUT})
endif()
