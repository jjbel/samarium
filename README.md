<!-- TODO make a script to merge these into 2x2 grid, loop them -->

<!-- gravity -->
[](https://github.com/user-attachments/assets/b461e8b8-eac3-4cb0-b3aa-b91b147f4b81)

<!-- flow field -->
[](https://github.com/user-attachments/assets/f8f23bc0-c454-481b-b536-9e6e7abb8bd9)

<!-- softbody -->
[](https://user-images.githubusercontent.com/83468982/178473002-b7f896f6-d5ed-4cc5-be34-bcccab9ef11e.mp4)

<!-- maxwell boltzmann -->
[](https://github.com/user-attachments/assets/1bd3aa90-80b7-486c-94df-1e03d7e276c9)

<!-- hilbert -->
<!-- [](https://user-images.githubusercontent.com/83468982/178472984-8cd83808-bfb2-478b-8a5e-3d45782f2c7d.mp4) -->

<!-- fourier India -->
<!-- [](https://github.com/user-attachments/assets/d870c975-44d4-4624-b122-48129506bbf6) -->

# Samarium

<!--
[![GCC](https://github.com/jjbel/samarium/actions/workflows/gcc.yml/badge.svg)](https://github.com/jjbel/samarium/actions/workflows/gcc.yml)
[![Clang](https://github.com/jjbel/samarium/actions/workflows/clang.yml/badge.svg)](https://github.com/jjbel/samarium/actions/workflows/clang.yml)
[![MSVC](https://github.com/jjbel/samarium/actions/workflows/msvc.yml/badge.svg)](https://github.com/jjbel/samarium/actions/workflows/msvc.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=jjbel_samarium&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=jjbel_samarium) -->


<!-- [![MIT License](https://img.shields.io/badge/license-MIT-yellow)](https://github.com/jjbel/samarium/blob/main/LICENSE.md) -->

<!--
![language: C++20](https://img.shields.io/badge/language-C%2B%2B20-yellow)
[![Latest Github Release](https://img.shields.io/github/v/tag/jjbel/samarium?label=latest%20release)](https://github.com/jjbel/samarium/tags) -->

Samarium is a 2d physics simulation and rendering library written in C++20, with a focus on high performance using GPU acceleration (CUDA and compute shaders), and by using the CPU and memory better (multithreading, data-oriented-design).

I am actively working on adding 3D support and more simulation domains: chemical reactions, phase change, cloth etc.

<!-- Rendering is done directly with OpenGL. -->
<!-- Offload more work to the GPU -->
<!-- SIMD for increasing, CPU performance -->

<!-- ## Contents -->

<!-- TODO use vscode markdown auto TOC -->

<!-- - [Examples](#examples) -->
<!-- - [Quickstart](#quickstart) -->
<!-- - [Prerequistes](#prerequistes)
- [Installation](#installation)
- [Example](#example)
- [Tools](#tools)
- [Documentation](#documentation)
- [License](#license) -->

## Quickstart

```sh
git clone --depth 1 https://github.com/jjbel/samarium.git
python samarium/bootstrap.py
```

<!-- TODO make sure bootstrap works -->
<!-- TODO make it easy to run examples, easier than copy pasting the code into a source file? -->

## Installation

<!-- | Dependency | URL                                 | Documentation               |
| ---------- | ----------------------------------- | --------------------------- |
| python     | <https://www.python.org/downloads/> |                             |
| git        | <https://git-scm.com/downloads/>    | <https://git-scm.com/docs/> | -->

<!-- | cmake      | <https://cmake.org/download/>       | <https://cmake.org/cmake/help/latest/> | -->
<!-- | conan      | <https://conan.io/downloads.html/> | <https://docs.conan.io/en/latest/> | -->

Install [python](https://www.python.org/downloads/) and [git](https://git-scm.com/docs/).

A compiler supporting C++20 is required, namely Visual C++ 2019, GCC-11, or Clang-13.

To install, do the following, or **just run `python samarium/boostrap.py`**
1. Install [CMake](https://cmake.org/download/) and [Conan](https://conan.io/downloads.html/)
2. Download [the zip](https://github.com/jjbel/samarium/archive/refs/heads/main.zip) or clone the repo:
```sh
git clone https://github.com/jjbel/samarium.git
```
3. build the library for your machine:
```sh
conan create ./samarium/ -b missing
```

<!-- ## Installation

To install the library locally:

```
conan download samarium/1.1.0@
```

or for the latest version

```sh
git clone --depth 1 https://github.com/jjbel/samarium.git
conan create ./samarium/ -b missing
``` -->

<!-- ## Example -->

For a fully-featured and self-contained example, run:

<!-- is depth 1 rly faster? -->

```sh
git clone --depth 1 https://github.com/jjbel/samarium_example.git .
cmake --preset default
cmake --build --preset default
```

### Now try the `examples/` directory!

## Documentation

View the docs at [Github Pages: https://jjbel.github.io/samarium/](https://jjbel.github.io/samarium/)

## License

Samarium is distributed under the [MIT License](LICENSE.md).

## Libraries Used

Many thanks to the following wonderful libraries:
1. [GLFW](https://www.glfw.org/), [glad](https://github.com/Dav1dde/glad) and [glm](https://github.com/g-truc/glm) for OpenGL support
2. [fmtlib](https://github.com/fmtlib/fmt) to completely replace `iostream`
3. [range-v3](https://github.com/ericniebler/range-v3)
4. [BS::thread_pool](https://github.com/bshoshany/thread-pool)
5. [PCG RNG](https://www.pcg-random.org/) for simple and fast randomness
6. [tl::function_ref](https://github.com/TartanLlama/function_ref) and [tl::expected](https://github.com/TartanLlama/expected)
7. [znone/call_thunk](https://github.com/znone/call_thunk) for using with GLFW's C callback
8. [itlib-static-vector](https://github.com/iboB/itlib)
9. [unordered-dense](https://github.com/martinus/unordered_dense) for faster `unordered_map`
10. [stb](https://github.com/nothings/stb) for image reading and writing
11. [FreeType](http://freetype.org/) for text rendering
