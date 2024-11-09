Building

CMake
presets
--build

cmake --build
I only use Release builds
For examples, all executables are in build/examples/ on Linux or build\\examples\\Release\\ on Windows. This is even if the source code is in a subfolder: eg examples/fourier/fourier.cpp is built to build/examples/fourier

Conan

All code is the namespace ``sm``
Code which clearly serves a distict purpose is put in a separate namespace, eg ``sm::math``, ``sm::interp``
Try to keep source code files < 400 lines of code each

Header only option

Naming Conventions
classes/structs: PascalCase
functions, namespaces: snake_case

modern C++
No pointers RAII if needed
no polymorphism
minimal abstraction. simple wrappers. eg ...
templates are useful
operator overloading for math types (Vector2 mainly), and callables 
sensible defaults. eg window size, escape on Esc

Helper Scripts
run.ps1
tasks.json, bind it to Numpad0, so can run with 1 keypress. Could use live reload, but not useful for GUI

examples, tests, benchmarks


For the optimal developing experience, use [VSCode](https://code.visualstudio.com) using the following extensions and tools

1. [C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack)
2. [Clang Format](https://clang.llvm.org/docs/ClangFormat.html)
3. [CMake Format](https://github.com/cheshirekow/cmake_format) and the corresponding [extension](https://marketplace.visualstudio.com/items?itemName=cheshirekow.cmake-format)
   <!-- 4. [SonarLint](https://marketplace.visualstudio.com/items?itemName=SonarSource.sonarlint-vscode) -->
   <!-- 5. [C++ Advanced Lint](https://marketplace.visualstudio.com/items?itemName=jbenden.c-cpp-flylint) -->
