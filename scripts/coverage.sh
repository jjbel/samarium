gcovr -r .. . -e "_deps/" --sonarqube -o ../coverage.xml
gcovr -r .. . -e "_deps/" --html -o ../coverage.html
gcovr -r .. . -e "_deps/"
