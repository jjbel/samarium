#!/bin/bash

cmake -B build >/dev/null \
&& mold -run cmake --build build -j 6 \
&& ./build/test/bin/samarium_tests
