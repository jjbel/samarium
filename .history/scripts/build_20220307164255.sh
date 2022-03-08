#!/bin/bash

mold -run cmake --build build -j 6 &> ./build/output
sed -i 's#home##g' ./build/output
sed -i 's#jb##g' ./build/output
sed -i 's#sm##g' ./build/output
sed -i 's#src##g' ./build/output
sed -i 's#samarium##g' ./build/output
sed -i 's#CMakeFiles##g' ./build/output
sed -i 's#.dir##g' ./build/output
sed -i 's#//##g' ./build/output
sed -i '#^/#d' ./build/output
cat ./build/output
exit $?
