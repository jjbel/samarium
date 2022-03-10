#!/bin/bash

find ./build -name "*.gcda" -type f -delete &> /dev/null
find ./build -name "*.gcno" -type f -delete &> /dev/null

./scripts/build.sh

# rm -f ./build/test/samarium_tests
# if [[ -f ./build/test/samarium_tests ]]
# then
    ./build/test/samarium_tests
# fi
