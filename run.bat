@echo off
@REM cls
cmake --build --preset=win && .\build\test\Release\samarium_tests
@REM cmake --build --preset=win && .\build\examples\Release\flow_field_noise
