#pragma once

#include "samarium/samarium.hpp"

namespace sm
{
struct Turtle
{
  public:
    Vector2f pos{};
    f32 angle                   = 0.0;
    static constexpr float size = 20;
    bool pen                    = true;

    std::vector<Vector2f> tri()
    {
        const auto disp = Vector2f(size, 0).rotated(angle);
        return {pos + disp, pos + disp.rotated(math::pi * 0.5),
                pos + disp.rotated(-math::pi * 0.5)};
    }

    void draw(Window& window)
    {
        draw::circle(window, {this->pos.cast<f64>(), 2}, {colors::red});
        // not drawing the circle gives a garbage triangle?

        draw::polygon(window, this->tri(), ShapeColor(colors::transparent, colors::red, 4),
                      window.view);
    }

    void left(f32 degrees) { this->angle += math::to_radians(degrees); }
    void right(f32 degrees) { this->angle -= math::to_radians(degrees); }
    void forward(f32 distance) { this->pos += Vector2f::from_polar({distance, this->angle}); }
};
} // namespace sm
