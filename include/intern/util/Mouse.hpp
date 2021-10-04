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

class Mouse // container for Mouse variables
{
public:
    bool wasPressed = false; // was the mouse pressed on the current frame
    bool wasMoved = false;   // was the mouse moved   on the current frame
    Vector2 pos;
    Vector2 posPrev;

    void update()
    {
        this->posPrev = this->pos; // update regardless of mouse events
    }
} mouse;

void handleMouseEvent(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouse.wasPressed = true;
        // cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        mouse.wasMoved = true;
        mouse.pos = Vector2(x, y);
        // cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
        // std::cout << "Moved by " << (pos - mousePosPrev).toString() << std::endl;
    }
    else
    {
        mouse.wasPressed = false;
        mouse.wasMoved = true;
    }
}