#!/bin/bash

cmake -B build >/dev/null
fd | entr -cc ./scripts/build.sh
