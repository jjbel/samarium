from conans import ConanFile, CMake


class SamariumConan(ConanFile):
    name = "samarium"
    version = "1.0"
    license = "MIT"
    author = "strangeQuark1041"
    url = "https://github.com/strangeQuark1041/samarium/"
    description = "2-D physics simulation engine"
    topics = ("c++20", "physics", "simulation")

    generators = "cmake"
    requires = "fmt/8.0.1", "sfml/2.5.1", "date/3.0.1", "range-v3/0.11.0", "boost/1.78.0"
    exports_sources = "src/*"
    options = {"testing": [True, False]}
    default_options = {"testing": False}

    def configure(self):
        self.options['sfml'].graphics = True
        self.options['sfml'].window = True
        self.options['sfml'].audio = False
        self.options['sfml'].network = False

    def requirements(self):
        if self.options.testing:
            self.requires('gtest/cci.20210126')

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
