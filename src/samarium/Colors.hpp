/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
 */

#pragma once

#include "Color.hpp"

namespace sm::colors
{
constexpr inline auto transparent          = Color::from_hex("#00000000");
constexpr inline auto black                = Color::from_hex("#000000");
constexpr inline auto silver               = Color::from_hex("#c0c0c0");
constexpr inline auto gray                 = Color::from_hex("#808080");
constexpr inline auto white                = Color::from_hex("#ffffff");
constexpr inline auto maroon               = Color::from_hex("#800000");
constexpr inline auto red                  = Color::from_hex("#ff0000");
constexpr inline auto purple               = Color::from_hex("#800080");
constexpr inline auto fuchsia              = Color::from_hex("#ff00ff");
constexpr inline auto green                = Color::from_hex("#008000");
constexpr inline auto lime                 = Color::from_hex("#00ff00");
constexpr inline auto olive                = Color::from_hex("#808000");
constexpr inline auto yellow               = Color::from_hex("#ffff00");
constexpr inline auto navy                 = Color::from_hex("#000080");
constexpr inline auto blue                 = Color::from_hex("#0000ff");
constexpr inline auto teal                 = Color::from_hex("#008080");
constexpr inline auto aqua                 = Color::from_hex("#00ffff");
constexpr inline auto orange               = Color::from_hex("#ffa500");
constexpr inline auto aliceblue            = Color::from_hex("#f0f8ff");
constexpr inline auto antiquewhite         = Color::from_hex("#faebd7");
constexpr inline auto aquamarine           = Color::from_hex("#7fffd4");
constexpr inline auto azure                = Color::from_hex("#f0ffff");
constexpr inline auto beige                = Color::from_hex("#f5f5dc");
constexpr inline auto bisque               = Color::from_hex("#ffe4c4");
constexpr inline auto blanchedalmond       = Color::from_hex("#ffebcd");
constexpr inline auto blueviolet           = Color::from_hex("#8a2be2");
constexpr inline auto brown                = Color::from_hex("#a52a2a");
constexpr inline auto burlywood            = Color::from_hex("#deb887");
constexpr inline auto cadetblue            = Color::from_hex("#5f9ea0");
constexpr inline auto chartreuse           = Color::from_hex("#7fff00");
constexpr inline auto chocolate            = Color::from_hex("#d2691e");
constexpr inline auto coral                = Color::from_hex("#ff7f50");
constexpr inline auto cornflowerblue       = Color::from_hex("#6495ed");
constexpr inline auto cornsilk             = Color::from_hex("#fff8dc");
constexpr inline auto crimson              = Color::from_hex("#dc143c");
constexpr inline auto cyan                 = Color::from_hex("#00ffff");
constexpr inline auto darkblue             = Color::from_hex("#00008b");
constexpr inline auto darkcyan             = Color::from_hex("#008b8b");
constexpr inline auto darkgoldenrod        = Color::from_hex("#b8860b");
constexpr inline auto darkgray             = Color::from_hex("#a9a9a9");
constexpr inline auto darkgreen            = Color::from_hex("#006400");
constexpr inline auto darkgrey             = Color::from_hex("#a9a9a9");
constexpr inline auto darkkhaki            = Color::from_hex("#bdb76b");
constexpr inline auto darkmagenta          = Color::from_hex("#8b008b");
constexpr inline auto darkolivegreen       = Color::from_hex("#556b2f");
constexpr inline auto darkorange           = Color::from_hex("#ff8c00");
constexpr inline auto darkorchid           = Color::from_hex("#9932cc");
constexpr inline auto darkred              = Color::from_hex("#8b0000");
constexpr inline auto darksalmon           = Color::from_hex("#e9967a");
constexpr inline auto darkseagreen         = Color::from_hex("#8fbc8f");
constexpr inline auto darkslateblue        = Color::from_hex("#483d8b");
constexpr inline auto darkslategray        = Color::from_hex("#2f4f4f");
constexpr inline auto darkslategrey        = Color::from_hex("#2f4f4f");
constexpr inline auto darkturquoise        = Color::from_hex("#00ced1");
constexpr inline auto darkviolet           = Color::from_hex("#9400d3");
constexpr inline auto deeppink             = Color::from_hex("#ff1493");
constexpr inline auto deepskyblue          = Color::from_hex("#00bfff");
constexpr inline auto dimgray              = Color::from_hex("#696969");
constexpr inline auto dimgrey              = Color::from_hex("#696969");
constexpr inline auto dodgerblue           = Color::from_hex("#1e90ff");
constexpr inline auto firebrick            = Color::from_hex("#b22222");
constexpr inline auto floralwhite          = Color::from_hex("#fffaf0");
constexpr inline auto forestgreen          = Color::from_hex("#228b22");
constexpr inline auto gainsboro            = Color::from_hex("#dcdcdc");
constexpr inline auto ghostwhite           = Color::from_hex("#f8f8ff");
constexpr inline auto gold                 = Color::from_hex("#ffd700");
constexpr inline auto goldenrod            = Color::from_hex("#daa520");
constexpr inline auto greenyellow          = Color::from_hex("#adff2f");
constexpr inline auto grey                 = Color::from_hex("#808080");
constexpr inline auto honeydew             = Color::from_hex("#f0fff0");
constexpr inline auto hotpink              = Color::from_hex("#ff69b4");
constexpr inline auto indianred            = Color::from_hex("#cd5c5c");
constexpr inline auto indigo               = Color::from_hex("#4b0082");
constexpr inline auto ivory                = Color::from_hex("#fffff0");
constexpr inline auto khaki                = Color::from_hex("#f0e68c");
constexpr inline auto lavender             = Color::from_hex("#e6e6fa");
constexpr inline auto lavenderblush        = Color::from_hex("#fff0f5");
constexpr inline auto lawngreen            = Color::from_hex("#7cfc00");
constexpr inline auto lemonchiffon         = Color::from_hex("#fffacd");
constexpr inline auto lightblue            = Color::from_hex("#add8e6");
constexpr inline auto lightcoral           = Color::from_hex("#f08080");
constexpr inline auto lightcyan            = Color::from_hex("#e0ffff");
constexpr inline auto lightgoldenrodyellow = Color::from_hex("#fafad2");
constexpr inline auto lightgray            = Color::from_hex("#d3d3d3");
constexpr inline auto lightgreen           = Color::from_hex("#90ee90");
constexpr inline auto lightgrey            = Color::from_hex("#d3d3d3");
constexpr inline auto lightpink            = Color::from_hex("#ffb6c1");
constexpr inline auto lightsalmon          = Color::from_hex("#ffa07a");
constexpr inline auto lightseagreen        = Color::from_hex("#20b2aa");
constexpr inline auto lightskyblue         = Color::from_hex("#87cefa");
constexpr inline auto lightslategray       = Color::from_hex("#778899");
constexpr inline auto lightslategrey       = Color::from_hex("#778899");
constexpr inline auto lightsteelblue       = Color::from_hex("#b0c4de");
constexpr inline auto lightyellow          = Color::from_hex("#ffffe0");
constexpr inline auto limegreen            = Color::from_hex("#32cd32");
constexpr inline auto linen                = Color::from_hex("#faf0e6");
constexpr inline auto magenta              = Color::from_hex("#ff00ff");
constexpr inline auto mediumaquamarine     = Color::from_hex("#66cdaa");
constexpr inline auto mediumblue           = Color::from_hex("#0000cd");
constexpr inline auto mediumorchid         = Color::from_hex("#ba55d3");
constexpr inline auto mediumpurple         = Color::from_hex("#9370db");
constexpr inline auto mediumseagreen       = Color::from_hex("#3cb371");
constexpr inline auto mediumslateblue      = Color::from_hex("#7b68ee");
constexpr inline auto mediumspringgreen    = Color::from_hex("#00fa9a");
constexpr inline auto mediumturquoise      = Color::from_hex("#48d1cc");
constexpr inline auto mediumvioletred      = Color::from_hex("#c71585");
constexpr inline auto midnightblue         = Color::from_hex("#191970");
constexpr inline auto mintcream            = Color::from_hex("#f5fffa");
constexpr inline auto mistyrose            = Color::from_hex("#ffe4e1");
constexpr inline auto moccasin             = Color::from_hex("#ffe4b5");
constexpr inline auto navajowhite          = Color::from_hex("#ffdead");
constexpr inline auto oldlace              = Color::from_hex("#fdf5e6");
constexpr inline auto olivedrab            = Color::from_hex("#6b8e23");
constexpr inline auto orangered            = Color::from_hex("#ff4500");
constexpr inline auto orchid               = Color::from_hex("#da70d6");
constexpr inline auto palegoldenrod        = Color::from_hex("#eee8aa");
constexpr inline auto palegreen            = Color::from_hex("#98fb98");
constexpr inline auto paleturquoise        = Color::from_hex("#afeeee");
constexpr inline auto palevioletred        = Color::from_hex("#db7093");
constexpr inline auto papayawhip           = Color::from_hex("#ffefd5");
constexpr inline auto peachpuff            = Color::from_hex("#ffdab9");
constexpr inline auto peru                 = Color::from_hex("#cd853f");
constexpr inline auto pink                 = Color::from_hex("#ffc0cb");
constexpr inline auto plum                 = Color::from_hex("#dda0dd");
constexpr inline auto powderblue           = Color::from_hex("#b0e0e6");
constexpr inline auto rosybrown            = Color::from_hex("#bc8f8f");
constexpr inline auto royalblue            = Color::from_hex("#4169e1");
constexpr inline auto saddlebrown          = Color::from_hex("#8b4513");
constexpr inline auto salmon               = Color::from_hex("#fa8072");
constexpr inline auto sandybrown           = Color::from_hex("#f4a460");
constexpr inline auto seagreen             = Color::from_hex("#2e8b57");
constexpr inline auto seashell             = Color::from_hex("#fff5ee");
constexpr inline auto sienna               = Color::from_hex("#a0522d");
constexpr inline auto skyblue              = Color::from_hex("#87ceeb");
constexpr inline auto slateblue            = Color::from_hex("#6a5acd");
constexpr inline auto slategray            = Color::from_hex("#708090");
constexpr inline auto slategrey            = Color::from_hex("#708090");
constexpr inline auto snow                 = Color::from_hex("#fffafa");
constexpr inline auto springgreen          = Color::from_hex("#00ff7f");
constexpr inline auto steelblue            = Color::from_hex("#4682b4");
constexpr inline auto tan                  = Color::from_hex("#d2b48c");
constexpr inline auto thistle              = Color::from_hex("#d8bfd8");
constexpr inline auto tomato               = Color::from_hex("#ff6347");
constexpr inline auto turquoise            = Color::from_hex("#40e0d0");
constexpr inline auto violet               = Color::from_hex("#ee82ee");
constexpr inline auto wheat                = Color::from_hex("#f5deb3");
constexpr inline auto whitesmoke           = Color::from_hex("#f5f5f5");
constexpr inline auto yellowgreen          = Color::from_hex("#9acd32");
constexpr inline auto rebeccapurple        = Color::from_hex("#663399");
} // namespace sm::colors
