# MIT License

# Copyright (c) 2022

# Project homepage: <https://github.com/strangeQuark1041/samarium/>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

# For more information, please refer to <https://opensource.org/licenses/MIT/>

from conans import ConanFile, CMake


class SamariumConan(ConanFile):
    name = "samarium"
    version = "1.0"
    license = "MIT"
    author = "strangeQuark1041"
    url = "https://github.com/strangeQuark1041/samarium/"
    description = "2-D physics simulation engine"
    topics = ("c++20", "physics", "2d", "simulation")

    generators = "cmake"
    requires = "fmt/8.0.1", "sfml/2.5.1", "date/3.0.1",
    exports_sources = "src/*"
    options = {"testing": [True, False]}
    default_options = {"testing": False}
    build_policy = "missing"

    def configure(self):
        self.options['sfml'].graphics = True
        self.options['sfml'].window = True
        self.options['sfml'].audio = False
        self.options['sfml'].network = False

    def requirements(self):
        if self.options.testing:
            self.requires('gtest/cci.20210126')
            self.requires('benchmark/1.6.0')

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="src")
        cmake.build()

    def package(self):
        self.copy("*.h*", dst="include", src="src")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["samarium"]
