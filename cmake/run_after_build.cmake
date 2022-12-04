if(RUN_AFTER_BUILD)
    add_custom_target(
        run ALL
        COMMAND ${RUN_AFTER_BUILD}
        COMMENT "Running '${RUN_AFTER_BUILD}'"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
endif()

add_dependencies(run samarium samarium_tests)
