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

#include "Bezier.hh"

namespace interp
{
    enum INTERP_TYPE
    {
        LINEAR,
        COSINE
    };

    inline double linear(double a, double b, double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        return a * (1 - t) + b * t;
    }

    inline Vector2 linear(Vector2 a, Vector2 b, double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        return a * (1 - t) + b * t;
    }

    inline double cosine(double a, double b, double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        double mu = (1 - std::cos(t * M_PI)) / 2;
        return a * (1 - mu) + b * mu;
    }

    inline Vector2 cosine(Vector2 a, Vector2 b, double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        double mu = (1 - std::cos(t * M_PI)) / 2;
        return a * (1 - mu) + b * mu;
    }

    inline double easeInSine(double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        return 1 - std::cos((t * M_PI) / 2);
    }

    double interp(double a, double b, double t, INTERP_TYPE type = LINEAR, bool clampFactor = true)
    {
        switch (type)
        {
        case COSINE:
            return cosine(a, b, t, clampFactor);

        default:
            return linear(a, b, t, clampFactor);
        }
    }

    Vector2 interp(Vector2 a, Vector2 b, double t, INTERP_TYPE type = LINEAR, bool clampFactor = true)
    {
        switch (type)
        {
        case COSINE:
            return cosine(a, b, t, clampFactor);

        default:
            return linear(a, b, t, clampFactor);
        }
    }

    inline double cubic(double y0, double y1, double y2, double y3, double t, bool clampFactor = true)
    {
        if (clampFactor)
            t = clamp<double>(t, 0, 1);
        double mu2 = t * t;
        double a0 = y3 - y2 - y0 + y1;
        double a1 = y0 - y1 - a0;
        double a2 = y2 - y0;
        double a3 = y1;

        return (a0 * t * mu2 + a1 * mu2 + a2 * t + a3);
    }
};