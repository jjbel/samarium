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
run.bat
tasks.json, bind it to Numpad0, so can run with 1 keypress. Could use live reload, but not useful for GUI

examples, tests, benchmarks
