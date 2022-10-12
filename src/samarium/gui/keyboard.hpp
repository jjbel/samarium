/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <functional> // for function
#include <utility>    // for move
#include <vector>     // for vector

#include "GLFW/glfw3.h"                     // for glfwGetKey, GLFWwindow
#include "range/v3/algorithm/all_of.hpp"    // for all_of, all_of_fn
#include "range/v3/functional/identity.hpp" // for identity

#include "samarium/core/types.hpp"        // for i32
#include "samarium/util/StaticVector.hpp" // for StaticVector

namespace sm
{
enum class Key
{
    Unknown        = GLFW_KEY_UNKNOWN,
    Space          = GLFW_KEY_SPACE,
    Apostrophe     = GLFW_KEY_APOSTROPHE, /* ' */
    Comma          = GLFW_KEY_COMMA,      /* , */
    Minus          = GLFW_KEY_MINUS,      /* - */
    Period         = GLFW_KEY_PERIOD,     /* . */
    Slash          = GLFW_KEY_SLASH,      /* / */
    Num0           = GLFW_KEY_0,
    Num1           = GLFW_KEY_1,
    Num2           = GLFW_KEY_2,
    Num3           = GLFW_KEY_3,
    Num4           = GLFW_KEY_4,
    Num5           = GLFW_KEY_5,
    Num6           = GLFW_KEY_6,
    Num7           = GLFW_KEY_7,
    Num8           = GLFW_KEY_8,
    Num9           = GLFW_KEY_9,
    Semicolon      = GLFW_KEY_SEMICOLON, /* ; */
    Equal          = GLFW_KEY_EQUAL,     /* = */
    A              = GLFW_KEY_A,
    B              = GLFW_KEY_B,
    C              = GLFW_KEY_C,
    D              = GLFW_KEY_D,
    E              = GLFW_KEY_E,
    F              = GLFW_KEY_F,
    G              = GLFW_KEY_G,
    H              = GLFW_KEY_H,
    I              = GLFW_KEY_I,
    J              = GLFW_KEY_J,
    K              = GLFW_KEY_K,
    L              = GLFW_KEY_L,
    M              = GLFW_KEY_M,
    N              = GLFW_KEY_N,
    O              = GLFW_KEY_O,
    P              = GLFW_KEY_P,
    Q              = GLFW_KEY_Q,
    R              = GLFW_KEY_R,
    S              = GLFW_KEY_S,
    T              = GLFW_KEY_T,
    U              = GLFW_KEY_U,
    V              = GLFW_KEY_V,
    W              = GLFW_KEY_W,
    X              = GLFW_KEY_X,
    Y              = GLFW_KEY_Y,
    Z              = GLFW_KEY_Z,
    LeftBracket    = GLFW_KEY_LEFT_BRACKET,  /* [ */
    Backslash      = GLFW_KEY_BACKSLASH,     /* \ */
    RightBracket   = GLFW_KEY_RIGHT_BRACKET, /* ] */
    GraveAccent    = GLFW_KEY_GRAVE_ACCENT,  /* ` */
    World1         = GLFW_KEY_WORLD_1,       /* non-US #1 */
    World2         = GLFW_KEY_WORLD_2,       /* non-US #2 */
    Escape         = GLFW_KEY_ESCAPE,
    Enter          = GLFW_KEY_ENTER,
    Tab            = GLFW_KEY_TAB,
    Backspace      = GLFW_KEY_BACKSPACE,
    Insert         = GLFW_KEY_INSERT,
    Delete         = GLFW_KEY_DELETE,
    Right          = GLFW_KEY_RIGHT,
    Left           = GLFW_KEY_LEFT,
    Down           = GLFW_KEY_DOWN,
    Up             = GLFW_KEY_UP,
    PageUp         = GLFW_KEY_PAGE_UP,
    PageDown       = GLFW_KEY_PAGE_DOWN,
    Home           = GLFW_KEY_HOME,
    End            = GLFW_KEY_END,
    CapsLock       = GLFW_KEY_CAPS_LOCK,
    ScrollLock     = GLFW_KEY_SCROLL_LOCK,
    NumLock        = GLFW_KEY_NUM_LOCK,
    PrintScreen    = GLFW_KEY_PRINT_SCREEN,
    Pause          = GLFW_KEY_PAUSE,
    F1             = GLFW_KEY_F1,
    F2             = GLFW_KEY_F2,
    F3             = GLFW_KEY_F3,
    F4             = GLFW_KEY_F4,
    F5             = GLFW_KEY_F5,
    F6             = GLFW_KEY_F6,
    F7             = GLFW_KEY_F7,
    F8             = GLFW_KEY_F8,
    F9             = GLFW_KEY_F9,
    F10            = GLFW_KEY_F10,
    F11            = GLFW_KEY_F11,
    F12            = GLFW_KEY_F12,
    F13            = GLFW_KEY_F13,
    F14            = GLFW_KEY_F14,
    F15            = GLFW_KEY_F15,
    F16            = GLFW_KEY_F16,
    F17            = GLFW_KEY_F17,
    F18            = GLFW_KEY_F18,
    F19            = GLFW_KEY_F19,
    F20            = GLFW_KEY_F20,
    F21            = GLFW_KEY_F21,
    F22            = GLFW_KEY_F22,
    F23            = GLFW_KEY_F23,
    F24            = GLFW_KEY_F24,
    F25            = GLFW_KEY_F25,
    Keypad0        = GLFW_KEY_KP_0,
    Keypad1        = GLFW_KEY_KP_1,
    Keypad2        = GLFW_KEY_KP_2,
    Keypad3        = GLFW_KEY_KP_3,
    Keypad4        = GLFW_KEY_KP_4,
    Keypad5        = GLFW_KEY_KP_5,
    Keypad6        = GLFW_KEY_KP_6,
    Keypad7        = GLFW_KEY_KP_7,
    Keypad8        = GLFW_KEY_KP_8,
    Keypad9        = GLFW_KEY_KP_9,
    KeypadDecimal  = GLFW_KEY_KP_DECIMAL,
    KeypadDivide   = GLFW_KEY_KP_DIVIDE,
    KeypadMultiply = GLFW_KEY_KP_MULTIPLY,
    KeypadSubtract = GLFW_KEY_KP_SUBTRACT,
    KeypadAdd      = GLFW_KEY_KP_ADD,
    KeypadEnter    = GLFW_KEY_KP_ENTER,
    KeypadEqual    = GLFW_KEY_KP_EQUAL,
    LeftShift      = GLFW_KEY_LEFT_SHIFT,
    LeftControl    = GLFW_KEY_LEFT_CONTROL,
    LeftAlt        = GLFW_KEY_LEFT_ALT,
    LeftSuper      = GLFW_KEY_LEFT_SUPER,
    RightShift     = GLFW_KEY_RIGHT_SHIFT,
    RightControl   = GLFW_KEY_RIGHT_CONTROL,
    RightAlt       = GLFW_KEY_RIGHT_ALT,
    RightSuper     = GLFW_KEY_RIGHT_SUPER,
    Menu           = GLFW_KEY_MENU,
    Last           = GLFW_KEY_LAST
};


namespace keyboard
{
using Action = std::function<void()>;
using KeySet = StaticVector<Key, 8>;

struct OnKeyPress
{
    GLFWwindow& window;
    KeySet key_set;
    Action action;

    explicit OnKeyPress(GLFWwindow& window_, KeySet key_set_, Action action_)
        : window{window_}, key_set(std::move(key_set_)), action(std::move(action_))
    {
    }

    void operator()() const;
};

struct OnKeyDown
{
    GLFWwindow& window;
    KeySet key_set;
    Action action;

    explicit OnKeyDown(GLFWwindow& window_, KeySet key_set_, Action action_)
        : window{window_}, key_set(std::move(key_set_)), action(std::move(action_))
    {
    }

    void operator()();

  private:
    bool previous{false};
};

struct OnKeyUp
{
    GLFWwindow& window;
    KeySet key_set;
    Action action;

    explicit OnKeyUp(GLFWwindow& window_, KeySet key_set_, Action action_)
        : window{window_}, key_set(std::move(key_set_)), action(std::move(action_))
    {
    }

    void operator()();

  private:
    bool previous{false};
};

class Keymap
{
    std::vector<Action> actions;

  public:
    explicit Keymap(std::vector<Action> event_listeners) : actions(std::move(event_listeners)) {}

    void push_back(const auto& action) { actions.emplace_back(action); }

    void clear();

    void run() const;
};
} // namespace keyboard
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_KEYBOARD_IMPL)

#include "range/v3/algorithm/all_of.hpp"
#include "range/v3/view/enumerate.hpp" // for enumerate

namespace sm::keyboard
{
void OnKeyPress::operator()() const
{
    if (ranges::all_of(key_set, [&](Key key)
                       { return glfwGetKey(&window, static_cast<i32>(key)) == GLFW_PRESS; }))
    {
        action();
    }
}

void OnKeyDown::operator()()
{
    const auto current = ranges::all_of(
        key_set, [&](Key key) { return glfwGetKey(&window, static_cast<i32>(key)) == GLFW_PRESS; });
    if (!previous && current) { action(); }
    previous = current;
}

void OnKeyUp::operator()()
{
    const auto current = ranges::all_of(
        key_set, [&](Key key) { return glfwGetKey(&window, static_cast<i32>(key)) == GLFW_PRESS; });
    if (!current && previous) { action(); }
    previous = current;
}

void Keymap::clear() { this->actions.clear(); }

void Keymap::run() const
{
    for (const auto& action : actions) { action(); }
}
} // namespace sm::keyboard

#endif
