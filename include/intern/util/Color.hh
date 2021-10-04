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

class Color
{
public:
    float red = 25;
    float green = 255;
    float blue = 255;
    float alpha = 255;

    Color(float red = 255.0f, float green = 255.0f, float blue = 255.0f, float alpha = 255.0f)
    {
        this->red = clamp(red, 0.0f, 1.0f);
        this->green = clamp(green, 0.0f, 1.0f);
        this->blue = clamp(blue, 0.0f, 1.0f);
        this->alpha = clamp(alpha, 0.0f, 1.0f);
    }

    Color(const Color &col)
    {
        this->red = col.red;
        this->green = col.green;
        this->blue = col.blue;
        this->alpha = col.alpha;
    }

    Color &operator=(const Color &b) //default assignment operator
    {
        return *this;
    }

    cv::Scalar toCvScalar()
    {
        return cv::Scalar(this->blue * 255, this->green * 255, this->red * 255, this->alpha * 255);
    }

    std::string toString() const
    {
        return fmt::format("Color({}, {}, {}, {})", this->red, this->green, this->blue, this->alpha);
        // return "Hello";
    }

    std::string ansiColor() const
    {
        return fmt::format("\x1b[38;2;{};{};{}m{}\x1b[0m", int(this->red * 255), int(this->green * 255), int(this->blue * 255), this->toString());
    }

    friend std::ostream &operator<<(std::ostream &out, const Color &c);

    friend std::ostream &operator<<(std::ostream &out, const Color &c);
};

namespace Colors
{
    const Color BLACK(0, 0, 0);
    const Color WHITE(255, 255, 255);
    const Color RED(255, 25, 40);
    const Color BLUE(80, 100, 200);
    const Color GREEN(10, 255, 10);
}

std::ostream &operator<<(std::ostream &os, const Color &col)
{
    os << col.ansiColor();
    return os;
}
