#!/bin/bash

echo "Configuring CMake..."
cmake -B build -G Ninja >/dev/null && \
fd | entr -cc ./scripts/run.sh
