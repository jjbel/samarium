#!/usr/bin/env python

from subprocess import run
from os import mkdir, environ
from sys import argv
from pathlib import Path

from format import format

CONFIGURE_COMMAND = 'cmake --preset=dev'
BUILD_COMMAND = 'cmake --build --preset=dev'
ROOT = Path(__file__).parent.parent
PRESET = argv[1] if len(argv) == 2 else 'dev'
FILTER_LIST = ['FAILED', 'isystem',
               'Building CXX object', 'ninja: build stopped', ' warnings and ']

if 'RELOAD_FILES' in environ:
    format(environ['RELOAD_FILES'].splitlines())
else:
    format() # format all files

if not (ROOT / 'build').exists():
    mkdir(ROOT / 'build')
    print('Running CMake...')
    run(CONFIGURE_COMMAND, shell=True)

root_str = str(ROOT.absolute())

print('Building...')
result = run(BUILD_COMMAND,
             shell=True, capture_output=True)
output = result.stdout.decode().replace(
    root_str, 'file://' + root_str).splitlines()


def is_valid(line: str):
    return not any(string in line for string in FILTER_LIST)


print('\n'.join(filter(is_valid, output)))
exit(result.returncode)
