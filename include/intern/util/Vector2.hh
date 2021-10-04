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
 * An implementation of 2-dimensional vectors
 * 
 * 
 * 
 */
class Vector2
{
public:
    double x;
    double y;
    double length_ = 0;
    double angle_ = 0;

    Vector2(double x = 0, double y = 0)
    {
        this->x = x;
        this->y = y;
    }

    Vector2(const Vector2 &vec)
    {
        this->x = vec.x;
        this->y = vec.y;
    }

    static Vector2 fromPolar(double length, double angle)
    {
        return Vector2(length * std::cos(angle), length * std::sin(angle));
    }

    std::string toString() const
    {
        return fmt::format("V[{}, {}]", this->x, this->y);
    }

    std::array<double, 2> toArray() const
    {
        return {this->x, this->y};
    }

    Vector2 copy() const
    {
        return Vector2(this->x, this->y);
    }

    double lengthSquared() const
    {
        return this->x * this->x + this->y * this->y;
    }

    double length() const
    {
        return std::sqrt(this->x * this->x + this->y * this->y);
    }

    double angle() const
    {
        return std::atan2(this->y, this->x);
    }

    // operator overloading:
    // all element wise
    Vector2 operator+(Vector2 const &obj)
    {
        return Vector2(this->x + obj.x, this->y + obj.y);
    }

    Vector2 operator+(double x)
    {
        return Vector2(std::cos(this->angle()) * (this->length() + x),
                       std::sin(this->angle()) * (this->length() + x));
    }

    Vector2 operator-(Vector2 const &obj)
    {
        return Vector2(this->x - obj.x, this->y - obj.y);
    }

    Vector2 operator-(double x)
    {
        return Vector2(std::cos(this->angle()) * (this->length() - x),
                       std::sin(this->angle()) * (this->length() - x));
    }

    Vector2 operator*(Vector2 const &obj)
    {
        return Vector2(this->x - obj.x, this->y - obj.y);
    }

    Vector2 operator*(double x)
    {
        return Vector2(std::cos(this->angle()) * (this->length() * x),
                       std::sin(this->angle()) * (this->length() * x));
    }

    Vector2 operator/(Vector2 const &obj)
    {
        return Vector2(this->x / obj.x, this->y / obj.y);
    }

    Vector2 operator/(double x)
    {
        return Vector2(std::cos(this->angle()) * (this->length() / x),
                       std::sin(this->angle()) * (this->length() / x));
    }

    Vector2 operator+=(Vector2 const &obj)
    {
        this->x += obj.x;
        this->y += obj.y;
        return *this;
    }

    Vector2 operator+=(double const &obj)
    {
        this->x = std::cos(this->angle()) * (this->length() + obj);
        this->y = std::sin(this->angle()) * (this->length() + obj);
        return *this;
    }

    Vector2 operator-=(Vector2 const &obj)
    {
        this->x -= obj.x;
        this->y -= obj.y;
        return *this;
    }

    Vector2 operator-=(double const &obj)
    {
        this->x = std::cos(this->angle()) * (this->length() - obj);
        this->y = std::sin(this->angle()) * (this->length() - obj);
        return *this;
    }

    Vector2 operator*=(Vector2 const &obj)
    {
        this->x *= obj.x;
        this->y *= obj.y;
        return *this;
    }

    Vector2 operator*=(double const &obj)
    {
        this->x *= obj;
        this->y *= obj;
        return *this;
    }

    Vector2 operator/=(Vector2 const &obj)
    {
        this->x /= obj.x;
        this->y /= obj.y;
        return *this;
    }

    Vector2 operator/=(double const &obj)
    {
        this->x /= obj;
        this->y /= obj;
        return *this;
    }

    bool operator==(double const &obj)
    {
        return (*this - obj).length() <= 0;
    }

    void operator=(double const &obj)
    {
        this->setLength(obj);
    }

    friend std::ostream &operator<<(std::ostream &out, const Vector2 &c);

    double dot(Vector2 vec) const
    {
        return this->x * vec.x + this->y * vec.y;
    }

    double perpDot(Vector2 vec) const
    {
        // https://www.xarg.org/book/linear-algebra/2d-perp-product/
        return this->x * vec.y - this->y * vec.x;
    }

    cv::Point toPoint() const
    {
        return cv::Point(this->x, this->y);
    }

    // for next functions, instead of void i return *this so I don't hv to write separate static functions

    Vector2 setX(double x)
    {
        this->x = x;

        return *this;
    }

    Vector2 setY(double y)
    {
        this->y = y;

        return *this;
    }

    Vector2 setXY(double x, double y)
    {
        this->x = x;
        this->y = y;

        return *this;
    }

    Vector2 setXY(Vector2 vec)
    {
        this->x = vec.x;
        this->y = vec.y;

        return *this;
    }

    Vector2 setAngle(double angle)
    {
        this->x = this->length() * std::cos(angle);
        return *this;
    }

    Vector2 setLength(double length)
    {
        // this->length() = std::abs(x);
        // if (x < 0)
        //     *this *= -1;
        // this->refreshXY();

        this->x = length * std::cos(this->angle());
        this->y = length * std::sin(this->angle());

        return *this;
    }

    Vector2 rotate(double x)
    {
        this->setAngle(fmod(this->angle() + x, 2 * M_PI)); // take mod 360deg so angle is in [0, 360]
        return *this;
    }

    Vector2 reflectSurface(Vector2 surface, double factor = 1) // reflect vector about normal
    {
        double tempAngle = surface.angle() - this->angle();
        this->rotate(tempAngle);
        *this *= factor;
        this->rotate(tempAngle);
        return *this;
    }

    Vector2 reflectNormal(Vector2 normal, double factor = 1) // reflect vector about normal of surface
    {
        double tempAngle = normal.angle() + (M_PI / 2) - this->angle();
        this->rotate(tempAngle);
        *this *= factor;
        this->rotate(tempAngle);
        return *this;
    }

    Vector2 normalize()
    {
        this->x /= this->length();
        this->y /= this->length();
        return *this;
    }

    Vector2 multiplyMatrix(double a_1_1, double a_1_2, double a_2_1, double a_2_2)
    {
        double temp = this->x; // x gets overwritten so keep it in temp
        this->x = a_1_1 * this->x + a_1_2 * this->y;
        this->y = a_2_1 * temp + a_2_2 * this->y;
        return *this;
    }

    void drawArrow(cv::Mat canvas, Vector2 pos, cv::Scalar color, double scale = 1, int thickness = 1, double arrowHeadShift = 1) const
    {
        // first part of arrow:
        cv::arrowedLine(canvas, pos.toPoint(), cv::Point(this->x * scale * arrowHeadShift + pos.x, this->y * scale * arrowHeadShift + pos.y), color, thickness, cv::LINE_AA);

        // line part of arrow
        // lerp between arrow and line, so arrowhead can be anywhere on arrow
        cv::line(canvas, cv::Point(this->x * scale * arrowHeadShift + pos.x, this->y * scale * arrowHeadShift + pos.y), cv::Point(this->x * scale + pos.x, this->y * scale + pos.y), color, thickness, cv::LINE_AA);
    }

    void drawLine(cv::Mat canvas, Vector2 pos, cv::Scalar color, double scale = 1, int thickness = 1) const
    {
        cv::line(canvas, pos.toPoint(), cv::Point(this->x * scale + pos.x, this->y * scale + pos.y), color, thickness, cv::LINE_AA);
    }

    // factor should belong to [0, 1]
    static Vector2 lerp(Vector2 a, Vector2 b, double factor)
    {
        return a * (1 - factor) + b * factor;
    }

    // project onto vec
    Vector2 project(Vector2 vec) const
    {
        Vector2 temp = vec.copy().normalize();
        return temp * this->dot(temp);
    }

    // project onto line
    Vector2 project(Vector2 a, Vector2 b)
    {
        return (*this - a).project(b - a) + a;
    }

    // projects p onto the line joining a and b, then calculates
    static double lerpInverse(Vector2 p, Vector2 a, Vector2 b)
    {
        double factor = (p.project(a, b).x - a.x) / (b.x - a.x);
        return std::isnan(factor) ? (p.project(a, b).y - a.y) / (b.y - a.y) : factor;
    }
};

std::ostream &operator<<(std::ostream &os, Vector2 &vec)
{
    os << vec.toString();
    return os;
}
