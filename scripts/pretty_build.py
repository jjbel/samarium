#!/usr/bin/env python

from subprocess import run
from os import mkdir
from sys import argv
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRESET = argv[1] if len(argv) == 2 else 'dev'
FILTER_LIST = ['FAILED', 'isystem',
               'Building CXX object', 'ninja: build stopped', ' warnings and ']

if not (ROOT/'build').exists():
    mkdir(ROOT/'build')
    print('Running CMake...')
    run('cmake --preset=dev', shell=True)

print('Building...')
result = run('cmake --build --preset=dev',
             shell=True, capture_output=True)
output = result.stdout.decode().replace(str(ROOT.absolute()), '').splitlines()


def is_valid(line: str):
    return not any(string in line for string in FILTER_LIST)


print('\n'.join(filter(is_valid, output)))
exit(result.returncode)
