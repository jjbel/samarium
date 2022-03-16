#!/bin/bash

cmake --preset=coverage
cmake --build --preset=coverage
./coverage.sh
