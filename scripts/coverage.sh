./test/samarium_tests

echo "Generating coverage..."
gcovr -r ../.. . -e "_deps/" -e "../../test/ut.hpp"  --sonarqube -o ../../coverage.xml
# gcovr -r ../.. . -e "_deps/" --html -o ../../coverage.html
gcovr -r ../.. . -e "_deps/" -e "../../test/ut.hpp" 
echo "Done"
