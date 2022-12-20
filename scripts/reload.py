#!/usr/bin/env python
from sys import argv
from os import chdir, name, environ, linesep
from subprocess import run
from time import sleep
from json import loads
from pathlib import Path


options = {
    'root_dir': Path(),  # current dir by default
    'commands': {},
    'delay_seconds': 0.001,
    'exit_on_error': False,
    'clear_screen': True,
    # set RELOAD_FILES: a list of changed files, separated by \n
    'set_files_env_var': True,
}


def git_ls(git_command_list):
    return [Path(x) for x in run(['git', 'ls-files'] + git_command_list, capture_output=True).stdout.decode()[:-1].splitlines()]


def get_files():
    return [i for i in sorted(git_ls([]) + git_ls(['-o', '--exclude-standard'])) if i not in git_ls(['-d'])]


def get():
    files = get_files()
    return files, {i: i.stat().st_mtime for i in files}


def run_command(command):
    run(command, shell=True, check=options['exit_on_error'])


def run_all():
    for command in options['commands']['all']:
        run_command(command)


def clear_screen():
    run('cls' if name == 'nt' else 'clear')


options_path = Path(__file__).parent / 'reload.json'
if len(argv) > 1:
    options_path = Path(argv[1])
options.update(loads(options_path.read_text()))
if not Path(options['root_dir']).is_dir():
    raise FileNotFoundError(f'{options["root_dir"]}: does not exist')

chdir(Path(options['root_dir']))
clear_screen()
run_all()
files, times = get()

try:
    while True:
        sleep(options['delay_seconds'])
        new_files, new_times = get()
        if new_times == times:
            continue
        if options['clear_screen']:
            clear_screen()
        # on directory structure changes
        if 'structure' in options['commands'] and new_files != files:
            for command in options['commands']['structure']:
                run_command(command)

        if 'set_files_env_var' in options and options['set_files_env_var']:
            environ['RELOAD_FILES'] = linesep.join([str(file) for file, time in set(
                new_times.items()).difference(times.items())])
        run_all()
        files = new_files
        times = new_times
except KeyboardInterrupt:
    print(' Exited.')  # extra space to separate from '^C'
