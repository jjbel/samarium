#!/bin/bash

set -e

cmake --preset=coverage >/dev/null
cmake --build --preset=coverage >/dev/null
ctest --preset=default
