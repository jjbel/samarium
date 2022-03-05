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
git clone https://github.com/strangeQuark1041/samarium.git
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

```sh
git clone https://github.com/strangeQuark1041/samarium.git
conan create samarium -b missing
```

## Example

In a new folder, create the following files:

`conanfile.txt`:

```Yaml
[requires]
samarium/1.0.0 # use samarium version 1.0.0 (the current stable version)

[generators]
cmake # use the default (and easiest to use) cmake config
```

`example.cpp`:

```cpp
#include "samarium/samarium.hpp"

int main()
{
    sm::print(sm::version); // sm::print calls fmt::print on all its args
    sm::print("A Vector2: ", sm::Vector2{.x = 5, .y = -3});
    sm::print("A Color:   ", sm::Color{.r = 5, .g = 200, .b = 255});
}
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(Example CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(example example.cpp)
target_link_libraries(example ${CONAN_LIBS})
```

Then run:
```sh
cmake -B build # cmake
cmake --build build # compile
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
