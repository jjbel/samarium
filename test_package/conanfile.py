import os

from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.build import cross_building
from conan.tools.build import can_run


class TestPackageConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    test_type = "explicit"

    def requirements(self):
        # TODO apparently have to propagate all deps
        # and in configure() below, their options too...

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

        for dep in deps:
            self.requires(dep)

        self.requires(self.tested_reference_str)

    def configure(self):
        self.options["glad"].gl_profile = "core"
        self.options["glad"].gl_version = "4.6"

        self.options["freetype"].with_png = False
        self.options["freetype"].with_zlib = False
        self.options["freetype"].with_bzip2 = False
        self.options["freetype"].with_brotli = False

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    # https://docs.conan.io/2/tutorial/creating_packages/test_conan_packages.html#tutorial-creating-test
    def test(self):
        if can_run(self):
            # # the executable name ie, the one in CMake add_executable
            # here it's test_package
            cmd = os.path.join(self.cpp.build.bindir, "test_package")
            self.run(cmd, env="conanrun")
