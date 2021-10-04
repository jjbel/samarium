/**
 * Copyright 2021. Jai Bellare
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <thread>

// #include "cxxopts-2.2.1/cxxopts.hpp"

int startup(int argc, char *argv[])
{
    system("clear;"); // for debugging

    std::string title = R"V0G0N(
   _____         __  __          _____  _____ _    _ __  __
  / ____|  /\   |  \/  |   /\   |  __ \|_   _| |  | |  \/  |
 | (___   /  \  | \  / |  /  \  | |__) | | | | |  | | \  / |
  \___ \ / /\ \ | |\/| | / /\ \ |  _  /  | | | |  | | |\/| |
  ____) / ____ \| |  | |/ ____ \| | \ \ _| |_| |__| | |  | |
 |_____/_/    \_\_|  |_/_/    \_\_|  \_\_____|\____/|_|  |_|
)V0G0N";

    for (auto i : title)
    {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::aquamarine), std::string(1, i));
        // std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    fmt::print(fmt::emphasis::bold | fg(fmt::color::alice_blue),
               "Version {}.{}\n", samarium_VERSION_MAJOR, samarium_VERSION_MINOR);

    fmt::print(fmt::emphasis::bold | fg(fmt::color::alice_blue), "Usage: ");
    fmt::print("{}", argv[0]);

    std::cout << "\n";

    return 0;
}

// int startup(int argc, char *argv[])
// {
//     cxxopts::Options options("samarium", "A 2d physics simulation package");

//     options.add_options()("s,scene", "Load scene from json file", cxxopts::value<std::string>())("d,debug", "Enable verbose debug printing", cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

//     auto result = options.parse(argc, argv);

//     if (result.count("help"))
//     {
//         std::cout << options.help() << std::endl;
//         exit(0);
//     }

//     bool debug = result["debug"].as<bool>();
//     std::string bar;
//     if (result.count("bar"))
//         bar = result["bar"].as<std::string>();

//     return 0;
// }