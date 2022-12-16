#!/usr/bin/env python

from subprocess import run
from os import chdir
from pathlib import Path

FORMATTER = 'clang-format'

print('running scripts/format.py...')

chdir(Path(__file__).parent.parent)
for file in run('git ls-files', shell=True, check=True, capture_output=True).stdout.decode().splitlines():
    if file.endswith('pp'):  # cpp, hpp
        run(['clang-format', '-i', file])
