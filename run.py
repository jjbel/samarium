from subprocess import run, Popen, PIPE
from sys import stdout as sys_stdout
from os import name
from pathlib import Path
from time import time

CONFIGURE_COMMAND = ["cmake", "--preset=win"]
BUILD_COMMAND = ["cmake", "--build", "--preset=win"]
RUN_COMMAND = ".\\build\\test\\Release\\samarium_tests.exe"

IGNORE_LINE_SUBSTRS = [
    "MSBuild",
    "Scanning sources",
    "samarium.vcxproj -> ",
    "samarium_tests.vcxproj -> ",
    "NVIDIA GPU Computing Toolkit",  # cuda compilation
    "Building Custom Rule",
]

REMOVE_SUBSTRS = [
    "D:\\sm\\samarium\\",
    "[build\\src\\samarium.vcxproj]",
    "[build\\test\\samarium_tests.vcxproj]",
]

# TODO regex
REPLACE_SUBSTRS = [(": error C", ": error ")]

# -----------------------------------------------------------------

# TODO how to run directly: .\run.py ? opens a new terminal window

# read stdout line-by-line in real time (?async)
# https://stackoverflow.com/a/42346942


class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def process_line(line):
    for substr in IGNORE_LINE_SUBSTRS:
        if substr in line:
            return False, ""

    for substr in REMOVE_SUBSTRS:
        line = line.replace(substr, "")

    for substr, replacement in REPLACE_SUBSTRS:
        line = line.replace(substr, replacement)

    # https://stackoverflow.com/q/287871
    if "error" in line:
        line = colors.FAIL + line + colors.ENDC
    elif "warning" in line:
        line = colors.WARNING + line + colors.ENDC

    return True, line


def build():
    proc = Popen(BUILD_COMMAND, stdout=PIPE)
    # , bufsize=1 removed: not in binary mode

    while True:
        line = proc.stdout.readline()
        if line:
            line = line.decode("utf-8")

            status, newline = process_line(line)

            if status:
                print(newline, end="")
            sys_stdout.flush()
        if proc.poll() is not None:  # process ends
            break

    return proc.wait()  # wait, and return returncode


def timed_build():
    start = time()
    returncode = build()
    end = time()
    print(f"Built in {end - start:.2f}s")
    return returncode


def run_exe():
    if Path(RUN_COMMAND).exists():
        run(RUN_COMMAND)


def main():
    if not Path("build").exists():
        run(CONFIGURE_COMMAND, check=True)

    run("cls" if name == "nt" else "clear", shell=True)

    if timed_build() == 0:
        run_exe()


main()
