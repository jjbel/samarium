#!/bin/bash

cmake -B build >/dev/null && \
fd | entr -cc ./scripts/run.sh

# echo "/home/jb/src/samarium/core/types.hpp /home/jb/src/samarium/core/concepts.hpp" | sed 's:/home/jb/src/samarium/::g'

