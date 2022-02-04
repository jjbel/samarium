#!/bin/bash

cmake --build build && \
./build/test/bin/samarium_tests
