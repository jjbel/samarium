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

#include <cmath>

constexpr double EPSILON = 1e-3;

// ********************************************************************************************************

std::array<Vector2, 2> boundingBox(Vector2 vec0, Vector2 vec1) // bounding box of 2 vectors, returns the top-left Vector and the bottom-right Vector
{
    std::array<Vector2, 2> vecs;
    vecs[0] = Vector2(std::min(vec0.x, vec1.x), std::min(vec0.y, vec1.y)); // min vector
    vecs[1] = Vector2(std::max(vec0.x, vec1.x), std::max(vec0.y, vec1.y)); // max vector
    return vecs;
}

std::array<Vector2, 2> boundingBox(Vector2 vec0, Vector2 vec1, Vector2 vec2, Vector2 vec3) // bounding box of 2 vectors, returns the top-left Vector and the bottom-right Vector
{
    std::array<Vector2, 2> vecs;
    vecs[0] = Vector2(std::min({vec0.x, vec1.x, vec2.x, vec3.x}), std::min({vec0.y, vec1.y, vec2.y, vec3.y})); // min vector
    vecs[1] = Vector2(std::max({vec0.x, vec1.x, vec2.x, vec3.x}), std::max({vec0.y, vec1.y, vec2.y, vec3.y})); // max vector
    return vecs;
}

bool inBoundingBox(Vector2 vec, Vector2 boxPoint0, Vector2 boxPoint1)
{
    // std::cout << "(boxPoint0.x - vec.x <= EPSILON) " << (boxPoint0.x - vec.x <= EPSILON) << "\n";
    // std::cout << "(vec.x - boxPoint1.x <= EPSILON) " << (vec.x - boxPoint1.x <= EPSILON) << "\n";
    // std::cout << "(boxPoint0.y - vec.y <= EPSILON) " << (boxPoint0.y - vec.y <= EPSILON) << "\n";
    // std::cout << "(vec.y - boxPoint1.y <= EPSILON) " << (vec.y - boxPoint1.y <= EPSILON) << "\n";
    return (boxPoint0.x - vec.x <= EPSILON) &&
           (vec.x - boxPoint1.x <= EPSILON) &&
           (boxPoint0.y - vec.y <= EPSILON) &&
           (vec.y - boxPoint1.y <= EPSILON);
}

bool doubleEquals(double a, double b, double epsilon)
{
    return std::abs(a - b) <= epsilon;
}

// from https://stackoverflow.com/a/32334103
bool nearly_equal(
    double a, double b,
    double epsilon = 128 * FLT_EPSILON, double abs_th = FLT_MIN)
// those defaults are arbitrary and could be removed
{
    if (!std::numeric_limits<double>::epsilon() <= epsilon)
        std::cout << "WARNING nearly_equal\n";
    if (!epsilon < 1.f)
        std::cout << "WARNING nearly_equal\n";

    if (a == b)
        return true;

    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<double>::max());
    // or even faster: std::min(std::abs(a + b), std::numeric_limits<double>::max());
    return diff < std::max(abs_th, epsilon * norm);
}

double doubleRand(double LO, double HI)
{
    return LO + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (HI - LO)));
}

double toRadians(double d)
{
    return (d * M_PI) / 180.0;
}

double toDegrees(double d)
{
    return d * 180.0 / M_PI;
}

// ! DOESN'T WORK
int pointOnSideOfLine(Vector2 testPoint, Vector2 linePoint0, Vector2 linePoint1)
{
    // https://math.stackexchange.com/questions/274712/clculate-on-which-side-of-a-straight-line-is-a-given-point-located
    Vector2 a = linePoint1 - linePoint0;
    Vector2 b = testPoint - linePoint0;
    double product = a.perpDot(b);
    if (std::abs(product) <= EPSILON)
        return 0;
    else if (product > 0)
        return 1;
    else
        return -1;
}

Vector2 slopeIntercept(Vector2 linePoint0, Vector2 linePoint1)
{
    double m = (linePoint1.y - linePoint0.y) / (linePoint1.x - linePoint0.x);
    double c = linePoint0.y - m * linePoint0.x;
    return Vector2(m, c);
}

// solve for the 2 intsersections of a line with a circle
std::array<Vector2, 2> *solveLineCircle(Vector2 centre, double radius, Vector2 linePoint0, Vector2 linePoint1)
{
    // https://1drv.ms/u/s!AvIZy0sNF870gQ1rnzGnfT_Rz3eI?e=HGfF8Z
    // TODO check for m -> infinity
    if (std::abs(linePoint1.x - linePoint0.x) <= EPSILON)
    {
        double x = linePoint0.x - centre.x;
        double term = sqrt((radius + x - centre.x) * (radius - x + centre.x));
        return new std::array<Vector2, 2>{
            Vector2(x, centre.y + term) + centre,
            Vector2(x, centre.y - term) + centre};
    }
    Vector2 line = slopeIntercept(linePoint0 - centre, linePoint1 - centre); // shift from centre of circle to origin
    double m = line.x, c = line.y;
    double sqrtTerm = (4 * m * m * c * c) - 4 * (m * m + 1) * (c * c - radius * radius);
    if (sqrtTerm < 0)
        return nullptr;
    double minusB = -2 * m * c;
    double sqrtedTerm = sqrt(sqrtTerm);
    double twoA = 2 * (m * m + 1);
    double x0 = (minusB + sqrtedTerm) / twoA;
    double x1 = (minusB - sqrtedTerm) / twoA;
    return new std::array<Vector2, 2>{Vector2(x0, m * x0 + c) + centre, Vector2(x1, m * x1 + c) + centre}; // reshift to centre

    // TODO implement bounding box
}

// express line joining points as  Vector2(m, c)

double distanceFromLine1(Vector2 point, Vector2 linePoint0, Vector2 linePoint1)
{
    return (point - linePoint0).length() * sin((point - linePoint0).angle() - (linePoint1 - linePoint0).angle());
}

double distanceFromLine(Vector2 point, Vector2 linePoint0, Vector2 linePoint1)
{
    return (point.project(linePoint0, linePoint1) - point).length();
}

bool inRange(double value, double min, double max, double epsilon)
{
    return min - epsilon <= value && value <= max + epsilon;
}

template<typename T = double>
T clamp(T value, T min, T max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

double maprange(double value, double from_min = 0, double from_max = 1, double to_min = 0, double to_max = 1, bool clamp = false)
{
    // https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    double fromRange = from_max - from_min;
    double toRange = to_max - to_min;
    double valueScaled = (value - from_min) / fromRange;
    return from_min + valueScaled * toRange;
}

Vector2 *solve(Vector2 _00, Vector2 _01, Vector2 _10, Vector2 _11, bool useBoundingBox = false)
/*
        solve for the intersection of 2 lines formed 4 vectors, optionally check if the solution is on the line segment and not just the lines, using a bounding box
    */
{
    double denominator0 = _01.x - _00.x;
    double denominator0Abs = abs(_01.x - _00.x);
    double denominator1 = _11.x - _10.x;
    double denominator1Abs = abs(_11.x - _10.x);
    double m0, m1, c0, c1, x, y;

    // first handle degenerate cases:
    if (denominator0Abs <= EPSILON && denominator1Abs > EPSILON) // line0 is vertical and not line1
    {
        x = _00.x; // don't find 0th slope as will be infinite
        m1 = (_11.y - _10.y) / denominator1;
        c1 = _10.y - m1 * _10.x;
        y = m1 * x + c1;
        // std::cout << "line0: vertical, line1: " << m1 << "\n";
    }
    else if (denominator1Abs <= EPSILON && denominator0Abs > EPSILON) // line1 is vertical and not line0
    {
        x = _10.x; // don't find 1st slope as will be infinite
        m0 = (_01.y - _00.y) / denominator0;
        c0 = _00.y - m1 * _00.x;
        y = m0 * x + c0;
        // std::cout << "line1: vertical, line0: " << m0 << "\n";
    }
    else if (
        denominator0Abs <= EPSILON && denominator1Abs <= EPSILON ||
        abs(
            (_01.y - _00.y) / denominator0 -
            (_11.y - _10.y) / denominator1) <= EPSILON) // both lines are vertical or within EPSILON of each other
    {
        std::cout << "both lines vertical or same slope" << std::endl;
        return nullptr;
    }
    else
    {
        m0 = (_01.y - _00.y) / denominator0;
        m1 = (_11.y - _10.y) / denominator1;
        c0 = _00.y - m0 * _00.x;
        c1 = _10.y - m1 * _10.x;
        x = (c1 - c0) / (m0 - m1);
        y = m0 * x + c0; // or m1 * x + c1
        // std::cout << "line0: " << m0 << ", line1: " << m1 << std::endl;
    }

    Vector2 *possibleSolution = new Vector2(x, y);

    if (!useBoundingBox)
    {
        std::cout << "Normal solution\n";
        return possibleSolution;
    }

    double t0 = Vector2::lerpInverse(*possibleSolution, _00, _01);
    double t1 = Vector2::lerpInverse(*possibleSolution, _10, _11);

    // std::cout << "inRange(t0, 0, 1, EPSILON) " << inRange(t0, 0, 1, EPSILON) << "\n";
    // std::cout << "inRange(t1, 0, 1, 4) " << inRange(t1, 0, 1, 4) << "\n";
    std::cout << "t0 " << t0 << "\n";
    std::cout << "t1 " << t1 << "\n";
    if (inRange(t0, 0, 1, EPSILON) && inRange(t1, 0, 1, EPSILON))
    {
        // std::cout << "In bounding box" << std::endl;
        return possibleSolution;
    }
    else
    {
        // std::cout << "not in bounding box" << std::endl;
        return nullptr;
    }
}