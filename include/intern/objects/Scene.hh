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

#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "util/Vector2.hh"
#include "util/math.hh"
#include "util/Mouse.hpp"
#include "util/Key.hh"
#include "util/interp.hh"

#include "objects/Object.hh"
#include "objects/Ball.hh"

using namespace std::chrono;

class Scene
{
private:
    uint16_t W;
    uint16_t H;
    uint16_t FPS;
    uint16_t SUBSTEPS;
    uint16_t SPEED;
    uint64_t frameCount;
    double dT;
    cv::Mat frame;
    bool running = true;

    std::string windowName = "OpenCV Image";

    const Vector2 GRAVITY = Vector2(0, 0);
    time_point<steady_clock> begin = steady_clock::now();

public:
    Scene(
        uint16_t width = 1280,
        uint16_t height = 720,
        uint16_t framerate = 24,
        uint16_t substeps = 1,
        uint16_t speed = 30)
    {
        this->W = width;
        this->H = height;
        this->FPS = framerate;
        this->SUBSTEPS = substeps;
        this->SPEED = speed;

        this->init();
    }

    void run()
    {
        for (; this->running; frameCount++)
        {
            this->begin = steady_clock::now();
            // break; // for debugging
            fmt::print("{:03}: ", frameCount);

            cv::setMouseCallback(windowName, handleMouseEvent, NULL);

            mouse.update();
            cv::imshow(windowName, frame);

            // processing uptil now takes time, so instead of naively delaying by the target fps, account for the time already taken
            float sleep_duration_ = 1000.0 / this->FPS;
            sleep_duration_ -= duration_cast<milliseconds>(steady_clock::now() - this->begin).count();
            sleep_duration_ = std::max(sleep_duration_, 1.0f);

            switch (cv::waitKey((int)(sleep_duration_)))
            {
            case Key::ESCAPE:
                running = false;
                break;
            }

            std::cout << std::endl;
        }
    }

private:
    void init()
    {
        this->dT = (double)this->SPEED / (this->SUBSTEPS * this->FPS);
        this->frameCount = 0;
        this->frame = cv::Mat(this->H, this->W, CV_8UC4);
    }
};