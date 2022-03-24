#!/bin/bash

echo "Configuring CMake..."
cmake --preset=home >/dev/null && \
echo "Done" && \
fd | entr -cc ./scripts/run.sh
