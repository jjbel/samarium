#pragma once

#include "samarium/graphics/colors.hpp"
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
    std::vector<LineSegment> segments{};

    Window window;

    Turtle(const WindowConfig& config = {{500, 500}, "Turtle Sim", false}) : window{config}
    {
        //  if window is 500x500 scale to pixel coords
        // 2.0 coz actually need to divide by 250
        // since default coords go from -1 to 1
        this->window.view.scale = Vector2(2.0, 2.0) / config.dims.cast<f64>();
    }

    std::vector<Vector2f> tri()
    {
        const auto disp = Vector2f(size, 0).rotated(angle);
        return {pos + disp, pos + disp.rotated(math::pi * 0.5),
                pos + disp.rotated(-math::pi * 0.5)};
    }

    void draw(f32 thickness = 3.0F)
    {
        for (const auto& seg : this->segments)
        {
            draw::line_segment(this->window, seg, Color{0, 0, 0}, thickness);
        }


        draw::circle(this->window, {this->pos.cast<f64>(), 2}, {colors::red});
        // not drawing the circle gives a garbage triangle?

        draw::polygon(this->window, this->tri(), ShapeColor(colors::red, colors::red, 4),
                      this->window.view);
    }

    void display()
    {
        this->draw();
        this->window.display();
        draw::background(colors::white);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // ----------------------------------------- API :

    void left(f32 degrees) { this->angle += math::to_radians(degrees); }
    void right(f32 degrees) { this->angle -= math::to_radians(degrees); }

    void forward(f32 distance)
    {
        const auto old_pos = this->pos;
        this->pos += Vector2f::from_polar({distance, this->angle});
        this->segments.push_back(LineSegment{old_pos.cast<f64>(), this->pos.cast<f64>()});

        this->display();
    }

    void getClick()
    {
        while (!this->window.mouse.left && this->window.is_open())
        {
            this->display();
            std::cout << this->window.mouse.left << std::endl;
        }
    }
};
} // namespace sm

#define main_program int main()

using namespace std;
using namespace sm;

static auto turtle = Turtle{};

void left(f32 degrees) { turtle.left(degrees); }
void right(f32 degrees) { turtle.right(degrees); }
void forward(f32 distance) { turtle.forward(distance); }
void getClick() { turtle.getClick(); }
