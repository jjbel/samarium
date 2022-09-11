#!/usr/bin/env python

from math import modf
from subprocess import run
from pathlib import Path
import time

SPHINXOPTS = []
SPHINXBUILD = "sphinx-build"
SOURCEDIR = "."
BUILDDIR = "build/html"

Path(BUILDDIR).mkdir(parents=True, exist_ok=True)

start = time.time()
run([SPHINXBUILD, "-b", "html", "-j", "auto", SOURCEDIR, BUILDDIR] + SPHINXOPTS)
end = time.time()
millis, seconds = modf(end - start)
print(f'Built in {int(seconds)}s {int(millis * 1000)}ms')
