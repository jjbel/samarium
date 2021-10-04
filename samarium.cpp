#include <iostream>
#include <memory>

#define FMT_HEADER_ONLY 1
#include <fmt/format.h>
#include <fmt/chrono.h>
#include <fmt/color.h>

#include "build/samariumConfig.h"
#include "startup/startup.hh"
#include "objects/Scene.hh"

int main(int argc, char *argv[])
{
    startup(argc, argv);

    Scene myscene;
    myscene.run();
}
