<h1 align="center">Samarium</h1>

<p align="center">
    <!-- <img alt="Github Action" src="https://github.com/strangeQuark1041/samarium/actions/workflows/build.yml/badge.svg"> -->
    <img alt="Lines of Code" src="https://img.shields.io/tokei/lines/github/strangeQuark1041/samarium">
    <img alt="Repo Size" src="https://img.shields.io/github/repo-size/strangeQuark1041/samarium">
    <img alt="Unlicense" src="https://img.shields.io/badge/License-Unlicense-lightgrey">
    <br>
    Samarium is an open-source 2d physics simulation package written in modern C++.
</p>

## Contents

- [Contents](#contents)
- [Quickstart](#quickstart)
- [Prerequistes](#prerequistes)
- [Installation](#installation)
- [Example](#example)
- [Documentation](#documentation)
- [Todo](#todo)
- [References](#references)

## Quickstart

```sh
git clone https://github.com/strangeQuark1041/samarium.git
conan create samarium
```

## Prerequistes

| Dependency | URL | Documentation |
| ---        | --- | --- |
| python     | <https://www.python.org/downloads/> | https://www.python.org/doc/ |
| git        | <https://git-scm.com/downloads> | https://git-scm.com/docs |
| cmake      | <https://cmake.org/download/> | https://cmake.org/cmake/help/latest/ |
| conan      | <https://conan.io/downloads.html> | https://docs.conan.io/en/latest/ |

**NOTE: A suitable C++ compiler which supports at least C++20 is required**

For example gcc11 at least.

## Installation

```sh
git clone https://github.com/strangeQuark1041/samarium.git
conan create samarium
```

## Example

In a new folder, called, for example **`myProj`**, create the following files:

`conanfile.txt`:

```Yaml
[requires]
samarium/1.0

[generators]
cmake
```

`example.cpp`:

```cpp
#include "samarium/Vector2.hpp"
#include "samarium/print.hpp"

int main()
{
    auto vec = sm::Vector2();
    sm::util::print("Samarium works! Print vector: ", vec);
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

Create a directory `build` and open your terminal in it:
```sh
conan install .. --build=missing # install dependencies
cmake .. # cmake
cmake --build . # compile
./bin/example # run the program
```

## Documentation

Documentation is located at the [Github wiki](https://github.com/strangeQuark1041/samarium/wiki) or locally in  [docs/README.md](docs/README.md)

## Todo

- add modules support

## References

These sources were of invaluable help during development:

1. C++ Standard: https://en.cppreference.com/
2. Custom iterators: https://internalpointers.com/post/writing-custom-iterators-modern-cpp
