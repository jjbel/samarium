#!/bin/bash

cmake -S . -B build >/dev/null && \
cmake --build build && \
./build/test/bin/samarium_tests_main
