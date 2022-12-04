if(RUN_AFTER_BUILD)
    add_custom_target(
        run ALL
        COMMAND ${RUN_AFTER_BUILD}
        COMMENT "Running '${RUN_AFTER_BUILD}'"
        DEPENDS ${RUN_AFTER_BUILD_DEPENDS}
        USES_TERMINAL
    )
    add_dependencies(run samarium samarium_tests)
endif()
