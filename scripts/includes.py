#!/usr/bin/env python

from pathlib import Path
from os import chdir


def base_name(path):
    return Path(path).name.split('.')[0].lower()


chdir(Path('src'))
files = list(Path('samarium').glob("**/*.hpp"))
skip = ['inline']
output = ""

for file in files:
    if base_name(file) in skip or str(file).count('/') != 2:
        continue

    for line in file.read_text().splitlines():
        if line.startswith('#include "samarium/'):
            included_path = line[10:]
            included_path = included_path[:included_path.index('"')]

            if base_name(included_path) in skip or included_path.count('/') != 2:
                continue

            output += f'"{str(included_path)[9:-4]}" -> "{str(file)[9:-4]}"\n'

with open(Path('../docs/src/build/includes.dot'), 'w') as file:
    file.write('''
digraph includes {
    layout="fdp";
    node [shape=box];
    smoothing="avg_dist";
    start=4;
''' + output + '}')
