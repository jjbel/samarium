#!/bin/bash

set -e

git checkout gh-pages

rm --force ./README.md

mkdir -p build
cd docs
doxygen
cd ..

cp -R ./build/docs/* .

git add .
git status
git commit -m "build docs"
