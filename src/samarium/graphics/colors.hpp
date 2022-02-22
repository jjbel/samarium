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
static constexpr auto transparent          = Color::from_hex("#00000000");
static constexpr auto black                = Color::from_hex("#000000");
static constexpr auto silver               = Color::from_hex("#c0c0c0");
static constexpr auto gray                 = Color::from_hex("#808080");
static constexpr auto white                = Color::from_hex("#ffffff");
static constexpr auto maroon               = Color::from_hex("#800000");
static constexpr auto red                  = Color::from_hex("#ff0000");
static constexpr auto purple               = Color::from_hex("#800080");
static constexpr auto fuchsia              = Color::from_hex("#ff00ff");
static constexpr auto green                = Color::from_hex("#008000");
static constexpr auto lime                 = Color::from_hex("#00ff00");
static constexpr auto olive                = Color::from_hex("#808000");
static constexpr auto yellow               = Color::from_hex("#ffff00");
static constexpr auto navy                 = Color::from_hex("#000080");
static constexpr auto blue                 = Color::from_hex("#0000ff");
static constexpr auto teal                 = Color::from_hex("#008080");
static constexpr auto aqua                 = Color::from_hex("#00ffff");
static constexpr auto orange               = Color::from_hex("#ffa500");
static constexpr auto aliceblue            = Color::from_hex("#f0f8ff");
static constexpr auto antiquewhite         = Color::from_hex("#faebd7");
static constexpr auto aquamarine           = Color::from_hex("#7fffd4");
static constexpr auto azure                = Color::from_hex("#f0ffff");
static constexpr auto beige                = Color::from_hex("#f5f5dc");
static constexpr auto bisque               = Color::from_hex("#ffe4c4");
static constexpr auto blanchedalmond       = Color::from_hex("#ffebcd");
static constexpr auto blueviolet           = Color::from_hex("#8a2be2");
static constexpr auto brown                = Color::from_hex("#a52a2a");
static constexpr auto burlywood            = Color::from_hex("#deb887");
static constexpr auto cadetblue            = Color::from_hex("#5f9ea0");
static constexpr auto chartreuse           = Color::from_hex("#7fff00");
static constexpr auto chocolate            = Color::from_hex("#d2691e");
static constexpr auto coral                = Color::from_hex("#ff7f50");
static constexpr auto cornflowerblue       = Color::from_hex("#6495ed");
static constexpr auto cornsilk             = Color::from_hex("#fff8dc");
static constexpr auto crimson              = Color::from_hex("#dc143c");
static constexpr auto cyan                 = Color::from_hex("#00ffff");
static constexpr auto darkblue             = Color::from_hex("#00008b");
static constexpr auto darkcyan             = Color::from_hex("#008b8b");
static constexpr auto darkgoldenrod        = Color::from_hex("#b8860b");
static constexpr auto darkgray             = Color::from_hex("#a9a9a9");
static constexpr auto darkgreen            = Color::from_hex("#006400");
static constexpr auto darkgrey             = Color::from_hex("#a9a9a9");
static constexpr auto darkkhaki            = Color::from_hex("#bdb76b");
static constexpr auto darkmagenta          = Color::from_hex("#8b008b");
static constexpr auto darkolivegreen       = Color::from_hex("#556b2f");
static constexpr auto darkorange           = Color::from_hex("#ff8c00");
static constexpr auto darkorchid           = Color::from_hex("#9932cc");
static constexpr auto darkred              = Color::from_hex("#8b0000");
static constexpr auto darksalmon           = Color::from_hex("#e9967a");
static constexpr auto darkseagreen         = Color::from_hex("#8fbc8f");
static constexpr auto darkslateblue        = Color::from_hex("#483d8b");
static constexpr auto darkslategray        = Color::from_hex("#2f4f4f");
static constexpr auto darkslategrey        = Color::from_hex("#2f4f4f");
static constexpr auto darkturquoise        = Color::from_hex("#00ced1");
static constexpr auto darkviolet           = Color::from_hex("#9400d3");
static constexpr auto deeppink             = Color::from_hex("#ff1493");
static constexpr auto deepskyblue          = Color::from_hex("#00bfff");
static constexpr auto dimgray              = Color::from_hex("#696969");
static constexpr auto dimgrey              = Color::from_hex("#696969");
static constexpr auto dodgerblue           = Color::from_hex("#1e90ff");
static constexpr auto firebrick            = Color::from_hex("#b22222");
static constexpr auto floralwhite          = Color::from_hex("#fffaf0");
static constexpr auto forestgreen          = Color::from_hex("#228b22");
static constexpr auto gainsboro            = Color::from_hex("#dcdcdc");
static constexpr auto ghostwhite           = Color::from_hex("#f8f8ff");
static constexpr auto gold                 = Color::from_hex("#ffd700");
static constexpr auto goldenrod            = Color::from_hex("#daa520");
static constexpr auto greenyellow          = Color::from_hex("#adff2f");
static constexpr auto grey                 = Color::from_hex("#808080");
static constexpr auto honeydew             = Color::from_hex("#f0fff0");
static constexpr auto hotpink              = Color::from_hex("#ff69b4");
static constexpr auto indianred            = Color::from_hex("#cd5c5c");
static constexpr auto indigo               = Color::from_hex("#4b0082");
static constexpr auto ivory                = Color::from_hex("#fffff0");
static constexpr auto khaki                = Color::from_hex("#f0e68c");
static constexpr auto lavender             = Color::from_hex("#e6e6fa");
static constexpr auto lavenderblush        = Color::from_hex("#fff0f5");
static constexpr auto lawngreen            = Color::from_hex("#7cfc00");
static constexpr auto lemonchiffon         = Color::from_hex("#fffacd");
static constexpr auto lightblue            = Color::from_hex("#add8e6");
static constexpr auto lightcoral           = Color::from_hex("#f08080");
static constexpr auto lightcyan            = Color::from_hex("#e0ffff");
static constexpr auto lightgoldenrodyellow = Color::from_hex("#fafad2");
static constexpr auto lightgray            = Color::from_hex("#d3d3d3");
static constexpr auto lightgreen           = Color::from_hex("#90ee90");
static constexpr auto lightgrey            = Color::from_hex("#d3d3d3");
static constexpr auto lightpink            = Color::from_hex("#ffb6c1");
static constexpr auto lightsalmon          = Color::from_hex("#ffa07a");
static constexpr auto lightseagreen        = Color::from_hex("#20b2aa");
static constexpr auto lightskyblue         = Color::from_hex("#87cefa");
static constexpr auto lightslategray       = Color::from_hex("#778899");
static constexpr auto lightslategrey       = Color::from_hex("#778899");
static constexpr auto lightsteelblue       = Color::from_hex("#b0c4de");
static constexpr auto lightyellow          = Color::from_hex("#ffffe0");
static constexpr auto limegreen            = Color::from_hex("#32cd32");
static constexpr auto linen                = Color::from_hex("#faf0e6");
static constexpr auto magenta              = Color::from_hex("#ff00ff");
static constexpr auto mediumaquamarine     = Color::from_hex("#66cdaa");
static constexpr auto mediumblue           = Color::from_hex("#0000cd");
static constexpr auto mediumorchid         = Color::from_hex("#ba55d3");
static constexpr auto mediumpurple         = Color::from_hex("#9370db");
static constexpr auto mediumseagreen       = Color::from_hex("#3cb371");
static constexpr auto mediumslateblue      = Color::from_hex("#7b68ee");
static constexpr auto mediumspringgreen    = Color::from_hex("#00fa9a");
static constexpr auto mediumturquoise      = Color::from_hex("#48d1cc");
static constexpr auto mediumvioletred      = Color::from_hex("#c71585");
static constexpr auto midnightblue         = Color::from_hex("#191970");
static constexpr auto mintcream            = Color::from_hex("#f5fffa");
static constexpr auto mistyrose            = Color::from_hex("#ffe4e1");
static constexpr auto moccasin             = Color::from_hex("#ffe4b5");
static constexpr auto navajowhite          = Color::from_hex("#ffdead");
static constexpr auto oldlace              = Color::from_hex("#fdf5e6");
static constexpr auto olivedrab            = Color::from_hex("#6b8e23");
static constexpr auto orangered            = Color::from_hex("#ff4500");
static constexpr auto orchid               = Color::from_hex("#da70d6");
static constexpr auto palegoldenrod        = Color::from_hex("#eee8aa");
static constexpr auto palegreen            = Color::from_hex("#98fb98");
static constexpr auto paleturquoise        = Color::from_hex("#afeeee");
static constexpr auto palevioletred        = Color::from_hex("#db7093");
static constexpr auto papayawhip           = Color::from_hex("#ffefd5");
static constexpr auto peachpuff            = Color::from_hex("#ffdab9");
static constexpr auto peru                 = Color::from_hex("#cd853f");
static constexpr auto pink                 = Color::from_hex("#ffc0cb");
static constexpr auto plum                 = Color::from_hex("#dda0dd");
static constexpr auto powderblue           = Color::from_hex("#b0e0e6");
static constexpr auto rosybrown            = Color::from_hex("#bc8f8f");
static constexpr auto royalblue            = Color::from_hex("#4169e1");
static constexpr auto saddlebrown          = Color::from_hex("#8b4513");
static constexpr auto salmon               = Color::from_hex("#fa8072");
static constexpr auto sandybrown           = Color::from_hex("#f4a460");
static constexpr auto seagreen             = Color::from_hex("#2e8b57");
static constexpr auto seashell             = Color::from_hex("#fff5ee");
static constexpr auto sienna               = Color::from_hex("#a0522d");
static constexpr auto skyblue              = Color::from_hex("#87ceeb");
static constexpr auto slateblue            = Color::from_hex("#6a5acd");
static constexpr auto slategray            = Color::from_hex("#708090");
static constexpr auto slategrey            = Color::from_hex("#708090");
static constexpr auto snow                 = Color::from_hex("#fffafa");
static constexpr auto springgreen          = Color::from_hex("#00ff7f");
static constexpr auto steelblue            = Color::from_hex("#4682b4");
static constexpr auto tan                  = Color::from_hex("#d2b48c");
static constexpr auto thistle              = Color::from_hex("#d8bfd8");
static constexpr auto tomato               = Color::from_hex("#ff6347");
static constexpr auto turquoise            = Color::from_hex("#40e0d0");
static constexpr auto violet               = Color::from_hex("#ee82ee");
static constexpr auto wheat                = Color::from_hex("#f5deb3");
static constexpr auto whitesmoke           = Color::from_hex("#f5f5f5");
static constexpr auto yellowgreen          = Color::from_hex("#9acd32");
static constexpr auto rebeccapurple        = Color::from_hex("#663399");
} // namespace sm::colors
