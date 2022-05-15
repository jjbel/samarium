#!/bin/bash

set -e

git checkout gh-pages

mkdir -p build
doxygen ./docs/Doxyfile

rm --force ./README.md

cp -R ./build/docs/* .

git add .
git commit -m "build docs" >/dev/null || 1
echo "Built docs"

git checkout main
