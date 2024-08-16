#!/usr/bin/env python

from subprocess import run, PIPE
from os import chdir
from pathlib import Path

try:
    import conan
    print('conan already installed.')
except ModuleNotFoundError:
    print('installing conan...')
    run('pip install --force-reinstall -v "conan==1.64"', shell=True, check=True, stdout=PIPE)
    print('done')

    # TODO test this on CI
    print('adding conan default profile...')
    run(['conan', 'profile', 'new', 'default', '--detect'],  stdout=PIPE)
    print('done')

chdir(Path(__file__).parent.parent.parent)

print('\ninstalling samarium (this may take some time to build dependencies)')
run(['conan', 'create', 'samarium',
    '-pr:b=default', '-b', 'missing'], check=True, stdout=PIPE)

print('\ncloning samarium_example into ./samarium_example')
run(['git', 'clone', '--depth', '1',
    'https://github.com/strangeQuark1041/samarium_example.git'], check=True)

chdir('samarium_example')
print('\nconfiguring example with "cmake --preset=default"')
run(['cmake', '--preset=default'], check=True)

print('\nbuilding example with "cmake --build --preset=default"')
run(['cmake', '--build', '--preset=default'], check=True)

# TODO only on windows
print('\nrunning turtle')
run(['run', 'turtle'], check=True)

print('\nExample ran successfully! Now check out the examples in the samarium/examples directory.')
