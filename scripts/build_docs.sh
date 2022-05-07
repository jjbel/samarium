#!/bin/bash

set -e

git checkout gh-pages

rm --force ./README.md

cmake --preset=default
cmake --build build --target docs

cp -R ./build/docs/* .

git add .
git status
git commit -m "build docs"
