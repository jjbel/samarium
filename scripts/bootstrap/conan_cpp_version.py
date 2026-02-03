from pathlib import Path
from subprocess import run

# https://docs.conan.io/2/reference/config_files/profiles.html
# default cpp version is that of compiler
# TODO do it using conan, not text editing?

def main():
    # strip removes the trailing newline
    path = Path(run('conan profile path default', shell=True, check=True, capture_output=True).stdout.decode().strip())
    text = path.read_text().splitlines()
    for index, line in enumerate(text):
        if line.startswith('compiler.cppstd='):
            text[index] = 'compiler.cppstd=20'
    path.write_text('\n'.join(text))
    print('set conan default profile cppstd=20')

if __name__ == '__main__':
    main()
