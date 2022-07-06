# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

from conans import ConanFile, CMake

required_conan_version = ">=1.43.0"


class SamariumConan(ConanFile):
    name = "samarium"
    version = "1.0.0"
    description = "2-D physics simulation library"
    homepage = "https://strangequark1041.github.io/samarium/"
    url = "https://github.com/conan-io/conan-center-index/"
    license = "MIT"
    topics = ("cpp20", "physics", "2d", "simulation")

    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [
        True, False], "build_tests": [True, False]}
    default_options = {"shared": False, "fPIC": True, "build_tests": False}

    generators = "cmake", "cmake_find_package"
    requires = "fmt/8.1.1", "sfml/2.5.1", "range-v3/0.12.0"
    exports_sources = "src/*"

    def requirements(self):
        if self.options.build_tests:
            self.requires('catch2/3.0.1')
            self.requires('benchmark/1.6.1')

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def build(self):
        cmake = CMake(self)  # get reference to cmake executable
        cmake.configure(source_folder="src")  # run cmake
        cmake.build()  # run cmake --build

    def package(self):
        self.copy("*.h*", dst="include", src="src")
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.dylib*", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["samarium"]
