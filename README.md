# Samarium

<!--
[![GCC](https://github.com/strangeQuark1041/samarium/actions/workflows/gcc.yml/badge.svg)](https://github.com/strangeQuark1041/samarium/actions/workflows/gcc.yml)
[![Clang](https://github.com/strangeQuark1041/samarium/actions/workflows/clang.yml/badge.svg)](https://github.com/strangeQuark1041/samarium/actions/workflows/clang.yml)
[![MSVC](https://github.com/strangeQuark1041/samarium/actions/workflows/msvc.yml/badge.svg)](https://github.com/strangeQuark1041/samarium/actions/workflows/msvc.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=strangeQuark1041_samarium&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=strangeQuark1041_samarium) -->

![Lines of Code](https://img.shields.io/endpoint?url=https://ghloc.vercel.app/api/jjbel/samarium/badge?filter=.hpp$,.cpp$,.glsl$&style=flat&logoColor=red&label=Lines%20of%20Code&color=red)
![Repo Size](https://img.shields.io/github/repo-size/strangeQuark1041/samarium)
[![MIT License](https://img.shields.io/badge/license-MIT-yellow)](https://github.com/strangeQuark1041/samarium/blob/main/LICENSE.md)

<!--
![language: C++20](https://img.shields.io/badge/language-C%2B%2B20-yellow)
[![Latest Github Release](https://img.shields.io/github/v/tag/strangeQuark1041/samarium?label=latest%20release)](https://github.com/strangeQuark1041/samarium/tags) -->

![MSVC build status](https://github.com/jjbel/samarium/actions/workflows/msvc.yml/badge.svg)
![MSVC build status](https://github.com/jjbel/samarium/actions/workflows/linux-gcc.yml/badge.svg)
![MSVC build status](https://github.com/jjbel/samarium/actions/workflows/linux-clang.yml/badge.svg)

Samarium is a 2d physics simulation library written in modern C++20.

## Contents
<!-- TODO use vscode markdown auto TOC -->

- [Examples](#examples)
- [Quickstart](#quickstart)
- [Prerequistes](#prerequistes)
- [Installation](#installation)
- [Example](#example)
- [Tools](#tools)
- [Documentation](#documentation)
- [License](#license)

## Examples

[](https://user-images.githubusercontent.com/83468982/178472984-8cd83808-bfb2-478b-8a5e-3d45782f2c7d.mp4)

[](https://user-images.githubusercontent.com/83468982/178473002-b7f896f6-d5ed-4cc5-be34-bcccab9ef11e.mp4)

## Quickstart

```sh
git clone --depth 1 https://github.com/strangeQuark1041/samarium.git
python samarium/bootstrap.py
```

## Prerequistes

| Dependency | URL                                 | Documentation               |
| ---------- | ----------------------------------- | --------------------------- |
| python     | <https://www.python.org/downloads/> |                             |
| git        | <https://git-scm.com/downloads/>    | <https://git-scm.com/docs/> |

<!-- | cmake      | <https://cmake.org/download/>       | <https://cmake.org/cmake/help/latest/> | -->
<!-- | conan      | <https://conan.io/downloads.html/> | <https://docs.conan.io/en/latest/> | -->

A compiler supporting C++20 is required, namely GCC-11, Clang-13, or Visual C++ 2019.

Running `boostrap.py` will install CMake and Conan, or you can do it yourself first.

## Installation

To install the library locally:

```
conan download samarium/1.1.0@
```

or for the latest version

```sh
git clone --depth 1 https://github.com/strangeQuark1041/samarium.git
conan create ./samarium/ -b missing
```

## Example

For a fully-featured and self-contained example, run:

```sh
git clone --depth 1 https://github.com/strangeQuark1041/samarium_example.git .
cmake --preset default
cmake --build --preset default
```

## Tools

For the optimal developing experience, use [VSCode](https://code.visualstudio.com) using the following extensions and tools

1. [C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
2. [Clang Format](https://clang.llvm.org/docs/ClangFormat.html)
3. [CMake Format](https://github.com/cheshirekow/cmake_format) and the corresponding [extension](https://marketplace.visualstudio.com/items?itemName=cheshirekow.cmake-format)
4. [SonarLint](https://marketplace.visualstudio.com/items?itemName=SonarSource.sonarlint-vscode)
5. [C++ Advanced Lint](https://marketplace.visualstudio.com/items?itemName=jbenden.c-cpp-flylint)

## Documentation

Documentation is located at [Github Pages](https://strangequark1041.github.io/samarium/)

## License

Samarium is distributed under the [MIT License](LICENSE.md).

Libraries used:

1. [fmtlib](https://github.com/fmtlib/fmt)
2. [range-v3](https://github.com/ericniebler/range-v3)
3. [BS::thread_pool](https://github.com/bshoshany/thread-pool)
4. [PCG RNG](https://www.pcg-random.org/)
5. [tl::function_ref](https://github.com/TartanLlama/function_ref)
