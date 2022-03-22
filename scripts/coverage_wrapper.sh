#!/bin/bash

rm -rf ./build
cmake --preset=coverage >/dev/null
cmake --build --preset=default >/dev/null
./scripts/coverage.sh
