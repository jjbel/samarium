from subprocess import run, PIPE
import conan_cpp_version


def main():
    try:
        import conan
        print('conan already installed.')
    except ModuleNotFoundError:
        # print('installing conan...')

        run('pip install conan', shell=True, check=True, capture_output=True)

        # print('adding conan default profile...')
        run(['conan', 'profile', 'detect', '--force'],  capture_output=True)

        print('installed conan and set default profile')

    # conan maybe already installed, but fix the cppstd anyway
    conan_cpp_version.main()


if __name__ == '__main__':
    main()
