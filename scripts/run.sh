#!/bin/bash

find ./build -name "*.gcda" -type f -delete
find ./build -name "*.gcno" -type f -delete

./scripts/build.sh && \
./build/test/samarium_tests
