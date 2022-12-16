#!/usr/bin/env python

from subprocess import run

for file in run('git ls-files', shell=True, check=True).stdout.decode().splitlines():
    run()
