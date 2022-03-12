#!/bin/bash

find ./build -name "*.gcda" -type f -delete &> /dev/null
find ./build -name "*.gcno" -type f -delete &> /dev/null
rm -f ./build/default/test/samarium_tests

./scripts/build.sh
~/bin/tryrun ./build/default/test/samarium_tests
