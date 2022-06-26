# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(SAMARIUM_SOURCES
    samarium/app/App.cpp
    samarium/util/ThreadPool.cpp
    samarium/graphics/Trail.cpp
    samarium/gui/Keyboard.cpp
    samarium/gui/Mouse.cpp
    samarium/physics/Spring.cpp
    samarium/physics/Particle.cpp
    samarium/physics/ParticleSystem.cpp
    samarium/physics/collision.cpp
    samarium/util/RandomGenerator.cpp
    samarium/util/Stopwatch.cpp
    samarium/util/file.cpp
    samarium/util/ostream.cpp
    samarium/util/util.cpp
    samarium/util/PerlinNoise.cpp
    PARENT_SCOPE
)
