#!/bin/bash

if [ ! -d "./build" ]; then
    cmake --preset=dev &> /dev/null
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
