#!/bin/bash

# ffmpeg -framerate 60 -pattern_type glob -i '*.tga' -pix_fmt yuv420p output.mp4
ffmpeg \
-t 6 \
-pattern_type glob -i '*.tga' \
-vf "fps=30,scale=200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
-loop 0 \
-y \
output.gif
