#!/usr/bin/env python

from subprocess import run
from os import chdir
from pathlib import Path
from sys import argv


def format(files=run('git ls-files', shell=True, check=True,
                     capture_output=True).stdout.decode().splitlines(), root_dir=Path(__file__).parent.parent, formatter='clang-format'):
    chdir(root_dir)

    for file in files:
        if file.endswith('pp'):  # cpp, hpp
            run([formatter, '-i', file])


if __name__ == 'main':
    if len(argv) >= 2:
        format(files=argv[1:])
    else:
        format()
