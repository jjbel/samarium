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
