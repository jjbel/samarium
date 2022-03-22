#!/bin/bash

 m -rf ./build
cmake --preset=coverage
cmake --build --preset=coverage
./scripts/coverage.sh
