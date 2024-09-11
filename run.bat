@echo off
@REM cls
cmake --build --preset=win && .\build\examples\Release\zoom
@REM cmake --build --preset=win && .\build\benchmarks\Release\vector_math
