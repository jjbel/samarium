@echo off
@REM cls
@REM cmake --build --preset=win && .\build\test\Release\samarium_tests
cmake --build --preset=win && .\build\examples\Release\fourier
