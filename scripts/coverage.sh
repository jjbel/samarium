#!/bin/bash

./build/test/samarium_tests

echo "Generating coverage..."
gcovr -r ./ ./build -e ./test/tests/ut.hpp
gcovr -r ./ ./build -e ./test/tests/ut.hpp --sonarqube -o ./coverage.xml
echo "Done"
