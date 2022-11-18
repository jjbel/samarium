#!/bin/bash

set -e

if [ ! -d "./build" ]; then
    echo "Running CMake..."
    cmake --preset=dev &> /dev/null && \
    echo "Done"
fi

PROGRAM=./build/test/samarium_tests
# PROGRAM=./build/benchmarks/samarium_benchmarks

rm -f ${PROGRAM}

echo "Compiling..."
./scripts/build.sh
echo "Done"

if test -f ${PROGRAM}
then
    ${PROGRAM}
fi

