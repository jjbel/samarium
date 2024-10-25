/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <string>
#include <vector>


#include "samarium/core/types.hpp"
#include "samarium/gl/draw/poly.hpp"
#include "samarium/gl/draw/shapes.hpp"
#include "samarium/graphics/Color.hpp"
#include "samarium/gui/Window.hpp"
#include "samarium/math/BoundingBox.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/util/unordered.hpp"

namespace sm
{
struct Plot
{
    using Float = f64;

    struct BoxStyle
    {
        bool draw     = true;
        Color color   = Color{255, 255, 255};
        f32 thickness = 0.01F;
    };

    struct Rescale
    {
        bool enabled       = true;
        f64 padding_factor = 1.1;
        // initial size (for zero/one point)
        // extra margin: relative or absolute
    };

    struct Axis
    {
        Float thickness{};
        bool draw           = true;
        bool draw_arrowhead = true;
        Rescale rescale{};

        // tick marks
        // numbers
        // use matplotlib naming conventions
    };

    struct Trace
    {
        Color color{255, 100, 0};
        f32 thickness = 0.014F;
        std::string display_name{};
        std::vector<Vector2f> points{};
    };

    // TODO gridlines

    BoundingBox<Float> box{};

    BoxStyle box_style{};
    Axis x_axis{};
    Axis y_axis{};
    Rescale rescale{};

    // scale of the plot. maps from plot space to window's world space
    Transform transform{};
    Map<std::string, Trace> traces{{"default", {}}};


    // Plot() {}

    // operator()(Float thickness, Color color) {}
    void add(const std::string& key, Vector2 point)
    {
        traces[key].points.push_back(point.cast<f32>());
    }
    void add(Vector2 point) { traces["default"].points.push_back(point.cast<f32>()); }

    void draw(Window& window)
    {
        if (rescale.enabled)
        {
            transform = Transform::map_boxes_from_to(
                bounding_box_plot_space().scaled(rescale.padding_factor), box);
        }

        // using window.zoom should not change line thickness
        // perhaps add a flag to disable this behavior
        const auto camera_scale_correction = 1.0F / static_cast<f32>(window.camera.scale.x);

        for (const auto& [key, trace] : traces)
        {
            if (trace.points.size() < 2) { continue; }
            // TODO this a potential bottleneck
            // the calculation of the vertices to draw the line segments
            // shoud be done in world space, else it looks squished
            // maybe add an overload for below fn which calls the transform in the loop only
            auto points = trace.points;
            for (auto& point : points) { point = transform(point); }

            draw::polyline_segments(window, points, trace.thickness * camera_scale_correction,
                                    trace.color);
        }

        if (box_style.draw)
        {
            draw::bounding_box(window, box, box_style.color,
                               box_style.thickness * camera_scale_correction);
        }
    }

  private:
    // TODO add default if no points / 1 point
    auto bounding_box_plot_space() const
    {
        using Box       = BoundingBox<f32>;
        auto box_       = Box{};
        auto flag_first = true;
        for (const auto& [key, trace] : traces)
        {
            if (trace.points.size() < 2) { continue; }
            const auto new_box = Box::fit_points(trace.points);
            if (flag_first) { box_ = new_box; }
            else { box_ = Box::fit_boxes(box_, new_box); }
            flag_first = false;
        }
        return box_.cast<f64>();
    }
};
} // namespace sm
