#!/bin/bash

if [ ! -d "./build" ]; then
    echo "Running CMake..."
    cmake --preset=dev &> /dev/null && \
    echo "Done"
fi

PROGRAM=./build/test/samarium_tests
# PROGRAM=./build/benchmarks/samarium_benchmarks

find ./build -name "*.gcda" -type f -delete &> /dev/null
find ./build -name "*.gcno" -type f -delete &> /dev/null
rm -f ${PROGRAM}

echo "Compiling..."
./scripts/build.sh
echo "Done"
~/bin/tryrun ${PROGRAM}
