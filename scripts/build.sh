#!/bin/bash

# cmake -B build >/dev/null && \
find ./build -name "*.gcda" -type f -delete
find ./build -name "*.gcno" -type f -delete

mold -run cmake --build build -j 6 && \
./build/test/samarium_tests
