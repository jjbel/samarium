from subprocess import run, CalledProcessError


def main():
    try:
        version_str = run('cmake --version', shell=True, check=True,
                          capture_output=True).stdout.decode().splitlines()[0]
        print(version_str, "already installed.")
    except CalledProcessError:
        run(['pip', 'install', 'cmake'], check=True, capture_output=True)
        print('Installed cmake')


if __name__ == '__main__':
    main()
