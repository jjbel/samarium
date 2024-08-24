# conan package manager https://conan.io/

from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout

required_conan_version = ">=2.6.0"


class SamariumConan(ConanFile):
    name = "samarium"
    version = "1.1.0"
    description = "2-D physics simulation library"
    homepage = "https://strangequark1041.github.io/samarium/"
    url = "https://github.com/conan-io/conan-center-index/"
    license = "MIT"
    topics = ("cpp20", "physics", "2d", "simulation")
    generators = "CMakeDeps", "CMakeToolchain"

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "header_only": [True, False],
        "build_tests": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "header_only": False,
        "build_tests": False,
    }

    exports_sources = "src/*"

    def requirements(self):
        deps = [
            "fmt/11.0.2",
            "range-v3/0.12.0",
            "stb/cci.20240531",
            "tl-expected/20190710",
            "tl-function-ref/1.0.0",
            "bshoshany-thread-pool/4.1.0",
            "unordered_dense/4.4.0",
            "svector/1.0.3",
            "glfw/3.4",
            "glm/cci.20230113",
            "glad/0.1.36",
            "freetype/2.13.2",
        ]

        if self.options.build_tests:
            deps += ["catch2/3.7.0", "benchmark/1.9.0"]

        for dep in deps:
            self.requires(dep)

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

        self.options["glad"].gl_profile = "core"
        self.options["glad"].gl_version = "4.6"

        self.options["freetype"].with_png = False
        self.options["freetype"].with_zlib = False
        self.options["freetype"].with_bzip2 = False
        self.options["freetype"].with_brotli = False

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        self.folders.source = "src"
        cmake_layout(self, src_folder="src")
        self.folders.generators = "build"

    def build(self):
        cmake = CMake(self)
        cmake.configure(
            variables={"SAMARIUM_HEADER_ONLY": str(self.options.header_only)}
        )
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["samarium"]
