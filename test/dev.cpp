/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

// #include "samarium/samarium.hpp"
#include "samarium/util/print.hpp"

#include "samarium/math/loop.hpp"

using namespace sm;
// using namespace sm::literals;

static constexpr auto src = R"glsl(
#version 460 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(r32f, binding = 0) uniform image2D out_tex;
void main() {
   // get position to read/write data from
   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
   // get value stored in the image
   float in_val = imageLoad( out_tex, pos ).r;
   // store new value in image
   imageStore( out_tex, pos, vec4( 69, 0.0, 0.0, 0.0 ) );
}
)glsl";

auto main() ->i32
{
    //    auto window       = Window{{{1800, 900}}};
    //    window.view.scale = Vector2::combine(1.0 / 10.0);
    //    auto point        = MovablePoint{{1.0, 1.0}, "#4e22ff"_c};
    //
    //    auto data = std::vector<f32>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    //    print("Data before:", data);
    //
    //    auto texture = gl::Texture{};
    //    texture.bind();
    //    texture.set_data(data, {data.size(), 1}, GL_R32F, GL_RED, GL_FLOAT);
    //    auto shader = expect(gl::ComputeShader::make(src));
    //    shader.bind(1, 1);
    //    glGetTextureImage(texture.handle, 0, GL_RED, GL_FLOAT, data.size() * sizeof(float),
    //                      data.data());
    //    print("Data after:", data);
    //
    //    const auto update = [&] { point.update(window.mouse); };
    //
    //    const auto draw = [&]
    //    {
    //        draw::background("#131417"_c);
    //        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 90}, .thickness =
    //        0.028F}); draw::circle(window, {{0.0, 0.0}, 0.1}, {.fill_color = "#ff0e4e"_c});
    //
    //        point.draw(window);
    //    };
    //
    //    run(window, update, draw);
    print("Start");
    for (auto i : loop::start_end<i32, loop::Interval::Closed>(15, 12)) { print(i); }
    print("End");
}
