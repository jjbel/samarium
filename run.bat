@echo off
@REM cls
cmake --build --preset=win
.\build\examples\Release\zoom
