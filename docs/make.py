#!/usr/bin/env python

from math import modf
from sys import argv
from subprocess import run
from pathlib import Path
import time

SOURCEDIR = "."
BUILDDIR = "build/html"
SPHINXOPTS = []
SPHINXBUILD = "sphinx-build"
TARGET = argv[1] if len(argv) == 2 else "dirhtml";

Path(BUILDDIR).mkdir(parents=True, exist_ok=True)

start = time.time()
run(
    [SPHINXBUILD, "-TEqb", TARGET, "-j", "auto", SOURCEDIR, BUILDDIR] + SPHINXOPTS,
    check=True,
)
end = time.time()

millis, seconds = modf(end - start)
print(f"Built in {int(seconds)}s {int(millis * 1000)}ms")
