#!/bin/bash

cmake --preset=coverage
cmake --build --preset=coverage
ctest --preset=default
