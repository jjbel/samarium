<h1 align="center">Samarium</h1>

<p align="center">
    <a href="https://github.com/strangeQuark1041/samarium/actions/workflows/build.yml">
         <img alt="CI status" src="https://github.com/strangeQuark1041/samarium/actions/workflows/build.yml/badge.svg">
    </a>
    <img alt="Lines of code" src="https://img.shields.io/tokei/lines/github/strangeQuark1041/samarium">
    <img alt="Repo Size" src="https://img.shields.io/github/repo-size/strangeQuark1041/samarium">
    <a href="https://github.com/strangeQuark1041/samarium/blob/main/LICENSE.md">
         <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-yellow">
    </a>
    <img alt="language: C++20" src="https://img.shields.io/badge/language-C%2B%2B20-yellow">
    <br>
    Samarium is a 2d physics simulation library written in modern C++20.
</p>

## Contents

- [Contents](#contents)
- [Quickstart](#quickstart)
- [Prerequistes](#prerequistes)
- [Installation](#installation)
- [Example](#example)
- [Documentation](#documentation)
- [License](#license)
- [Todo](#todo)
- [References](#references)

## Quickstart

```sh
pip install conan
git clone --depth 1 https://github.com/strangeQuark1041/samarium.git
conan create samarium -b missing
```

## Prerequistes

| Dependency | URL | Documentation |
| ---        | --- | --- |
| python     | <https://www.python.org/downloads/> | https://www.python.org/doc/ |
| git        | <https://git-scm.com/downloads/> | https://git-scm.com/docs/ |
| cmake      | <https://cmake.org/download/> | https://cmake.org/cmake/help/latest/ |
| conan      | <https://conan.io/downloads.html/> | https://docs.conan.io/en/latest/ |

Compiler wise, at least gcc-11 or Visual C++ 2019 is required
Clang [does not yet support C++20](https://clang.llvm.org/cxx_status.html#cxx20)

## Installation

To install the library locally:

```sh
git clone --depth 1 https://github.com/strangeQuark1041/samarium.git
conan create samarium -b missing
```

## Example

For a fully-featured and self-contained example, run:

```sh
git clone --depth 1 https://github.com/strangeQuark1041/samarium_example.git .
conan install . -b missing -if ./build # Install deps in build folder
cmake -B ./build
cmake --build ./build
./build/bin/example
```

## Documentation

Documentation is located on [Github Pages here](https://strangequark1041.github.io/samarium/) or locally in  [docs/README.md](docs/README.md)

## License

Samarium is distributed under the permissive [MIT License](LICENSE.md).

## Todo

- make Rendering multithreaded
- refactor, Color, Gradient handling into Material
- specify bounding box for `Renderer::draw_line`
- add modules support

## References

These sources were of invaluable help during development:

1. C++ Standard: https://en.cppreference.com/
2. Custom iterators: https://internalpointers.com/post/writing-custom-iterators-modern-cpp
