#!/bin/bash

find ./build -name "*.gcda" -type f -delete
find ./build -name "*.gcno" -type f -delete
rm -f ./build/test/samarium_tests

./scripts/build.sh && \
./build/test/samarium_tests
