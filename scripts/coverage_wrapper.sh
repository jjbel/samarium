#!/bin/bash

cmake --preset=coverage
cmake --build --preset=coverage
./scripts/coverage.sh
