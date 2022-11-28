#!/bin/bash

set -e

if [ ! -d "./build" ]; then
    echo "Running CMake..."
    cmake --preset=dev &> /dev/null && \
    echo "Done"
fi

echo "Building..."
./scripts/build.sh
