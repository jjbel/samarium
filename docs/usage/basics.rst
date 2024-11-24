Basics
======

.. code-block:: cpp

    // Include all the things
    #include "samarium/samarium.hpp"

    // All code is in namespace sm
    using namespace sm;

    auto main() -> i32
    {
        // print calls fmt::format() on each argument
        print("Hello there");

        // For uniformity, typedefs are used for numeric types
        f64 this_is_a_double = 3.14;
        f32 this_is_a_float = 3.14F;
        i32 this_is_an_int = 42;

        // A Vec2 is a pair of f64's: an x and y coordinate
        // it is an alias for Vec2_t<f64>
        print("A Vec2:", Vec2{.x = 5, .y = -3});

        // A Color is 4 u8's: red, green blue, and alpha
        print("A Color:  ", Color{.r = 5, .g = 200, .b = 255});

        // Color 
    }
