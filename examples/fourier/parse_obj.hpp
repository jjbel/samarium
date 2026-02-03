#pragma once

#include <filesystem>
#include <ranges>
#include <string>

#include "range/v3/algorithm/starts_with.hpp"

#include "samarium/math/Vec2.hpp"

auto read_file(const std::filesystem::path& path)
{
    // ifstream doesn't check if the path exists
    // TODO could use sm::Error
    if (!std::filesystem::exists(path)) { throw std::runtime_error("path does not exist"); }

    // needs to be non-const
    auto stream = std::ifstream{path};
    return std::string(std::istreambuf_iterator<char>{stream}, {});
}

// export obj's from Blender
auto obj_to_pts(const std::filesystem::path& obj_file)
{
    const auto line_to_vec = [](const auto& line)
    {
        // TODO why using std::ranges::to here, but ranges::to below
        // in rocket-ui std::ranges gave error
        const auto components =
            std::string(line.begin() + 2, line.end()) | std::views::split(' ') |
            std::views::transform([](auto r) { return std::string(r.data(), r.size()); }) |
            ranges::to<std::vector>();

        // blender needs y-axis flipping?
        return sm::Vec2{std::stod(components[0]), -std::stod(components[2])};
    };

    // TODO doesn't work with range-v3?
    // TODO copies? references?
    return read_file(obj_file) | std::views::split('\n') |
           std::views::transform([](auto r) { return std::string(r.data(), r.size()); }) |
           std::views::filter([](const auto& line) { return line.starts_with("v "); }) |
           std::views::transform(line_to_vec) | ranges::to<std::vector>();
}
