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

class CubicBezier
{
public:
    Vector2 p0;
    Vector2 p1;
    Vector2 p2;
    Vector2 p3;

    CubicBezier(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3)
    {
        this->p0 = p0;
        this->p1 = p1;
        this->p2 = p2;
        this->p3 = p3;
    }

    Vector2 at(double t)
    {
        return this->p0 * (-t * t * t + 3 * t * t - 3 * t + 1) +
               this->p1 * (3 * t * t * t - 6 * t * t + 3 * t) +
               this->p2 * (-3 * t * t * t + 3 * t * t) +
               this->p3 * (t * t * t);
    }

    Vector2 operator()(double t)
    {
        return this->at(t);
    }

    void draw(cv::Mat canvas, cv::Scalar color = cv::Scalar(220, 100, 255), int thickness = 1, int steps = 10)
    {
    }

    std::string toString()
    {
        std::ostringstream buffer;
        buffer << "Bezier: p0: " << this->p0 << ", p1: " << this->p1 << ", p2: " << this->p2 << ", p3: " << this->p3 << "\n";
        return buffer.str();
    }
};

std::ostream &operator<<(std::ostream &os, CubicBezier &curve)
{
    os << curve.toString();
    return os;
}
