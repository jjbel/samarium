./test/samarium_tests

echo "Generating coverage..."
gcovr -r .. . -e "_deps/" --sonarqube -o ../coverage.xml
gcovr -r .. . -e "_deps/" --html -o ../coverage.html
gcovr -r .. . -e "_deps/"
echo "Done"
