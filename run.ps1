function Cmake-Build {
    # TODO how to do live, ie streaming the lines, and not evaluating the entire expression
    # find a program which does it, or write on in C++
    # $cm_out =
    (cmake --build --preset=win `
    | Select-String -NoEmphasis -Pattern 'error|warning' `
    | Out-String).replace('D:\sm\samarium\', '').replace(' [build\test\samarium_tests.vcxproj]', '')
}

# TODO runs even on compile failure
Cmake-Build && .\build\test\Release\samarium_tests