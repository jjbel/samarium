#!/bin/bash

set -e

git checkout gh-pages

rm ./.md

cmake --preset=default
cmake --build build --target docs

cp -R ./build/docs/* .

git status
