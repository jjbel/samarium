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

/**
 * Circle
 * 
 * 
 * 
 */
class Ball
{
public:
    Color color;
    Vector2 pos;
    Vector2 posPrev;
    Vector2 vel;
    Vector2 acc;
    double radius;
    double mass;

    Ball(
        Color color = Colors::RED,
        double radius = 40,
        Vector2 pos = Vector2(0, 0),
        Vector2 vel = Vector2(0, 0))
    {
        this->pos = pos;
        this->vel = vel;
        this->pos = pos;
    }

    // Ball(const Ball &vec)
    // {
    //     this->x = vec.x;
    //     this->y = vec.y;
    // }

    // std::string toString() const
    // {
    //     return fmt::format("V[{}, {}]", this->x, this->y);
    // }

    // Ball copy() const
    // {
    //     return Ball(this->x, this->y);
    // }

    // friend std::ostream &operator<<(std::ostream &out, const Ball &c);

    // Vector2 reflectSurface(Vector2 surface, double factor = 1) // reflect vector about normal
    // {
    //     double tempAngle = surface.angle() - this->angle();
    //     this->rotate(tempAngle);
    //     *this *= factor;
    //     this->rotate(tempAngle);
    //     return *this;
    // }

    // Vector2 reflectNormal(Vector2 normal, double factor = 1) // reflect vector about normal of surface
    // {
    //     double tempAngle = normal.angle() + (M_PI / 2) - this->angle();
    //     this->rotate(tempAngle);
    //     *this *= factor;
    //     this->rotate(tempAngle);
    //     return *this;
    // }
};

// std::ostream &operator<<(std::ostream &os, Ball &ball)
// {
//     os << ball.toString();
//     return os;
// }
