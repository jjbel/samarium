#!/bin/bash

cmake -B build -G Ninja >/dev/null && \
fd | entr -cc ./scripts/run.sh
