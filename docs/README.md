# Samarium Documentation

- [Samarium Documentation](#samarium-documentation)
  - [Development](#development)
  - [Modules](#modules)

## Development

- Samarium uses [`git`](https://git-scm.com/) for version control, [`conan`](https://conan.io/) for package management and [`cmake`](https://cmake.org/) as a build generator
- Dependencies do not need to be installed manually
- By default, `cmake` builds an executable `samarium` in the `bin` directory
- Note - use your editor's clang-format plugin if it exists, for proper formatting
- The actual source code is located in `src/lib`, which builds a static library `libsm.a`
- This links with a file defining `int main()`, `src/main.cpp` by default.
- The root `CMakeLists.txt` merely adds the subdirectories `src` and `test`, and enables development specific options like testing and compiler flags.

## Modules

Samarium consists of the folowing modules:

1. `math`: core math functions and computation
2. `graphics`: `Color`, `Image`, and `Renderer`
