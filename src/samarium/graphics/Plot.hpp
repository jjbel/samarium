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
#include "samarium/gl/Text.hpp"
#include "samarium/gl/draw/poly.hpp"
#include "samarium/gl/draw/shapes.hpp"
#include "samarium/graphics/Color.hpp"
#include "samarium/gui/Window.hpp"
#include "samarium/math/Box2.hpp"
#include "samarium/math/Vec2.hpp"
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
        f32 thickness = 0.003F;
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
        std::vector<Vec2f> points{};
    };

    struct Title
    {
        std::string text{};
        Color color = {255, 255, 255};
        f32 scale   = 0.05F;
    };

    // TODO gridlines

    Box2<Float> box{};

    BoxStyle box_style{};
    Axis x_axis{};
    Axis y_axis{};
    Rescale rescale{};

    // scale of the plot. maps from plot space to window's world space
    Transform transform{};
    Map<std::string, Trace> traces{{"default", {}}};
    Title title{};
    draw::Text text{};

    Plot(const std::string& font = "CascadiaCode.ttf", u32 font_pixel_height = 96)
    {
        text = expect(draw::Text::make(font, font_pixel_height));
    }

    void add(const std::string& key, Vec2 point)
    {
        traces[key].points.push_back(point.template cast<f32>());
    }
    void add(Vec2 point) { traces["default"].points.push_back(point.template cast<f32>()); }

    void draw(Window& window)
    {
        if (rescale.enabled)
        {
            // TODO scaling to fit text. make this editable, calculate it from text size...
            transform = Transform::map_boxes_from_to(
                bounding_box_plot_space().scaled(rescale.padding_factor), box.scaled_y(0.8));
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

        if (!title.text.empty())
        {
            const auto box_          = transform.apply(box);
            constexpr auto placement = Placement{PlacementX::Middle, PlacementY::Top};
            const auto position      = box.get_placement(placement);
            text(window, title.text, position.template cast<f32>(),
                 title.scale * camera_scale_correction, title.color, placement);
        }
    }

  private:
    // TODO add default if no points / 1 point
    Box2<f64> bounding_box_plot_space() const
    {
        using Box       = Box2<f32>;
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
        return box_.template cast<f64>();
    }
};
} // namespace sm
