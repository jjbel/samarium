/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

// #include "samarium/graphics/colors.hpp"
#include "samarium/cuda/HostDevVec.hpp"
#include "samarium/samarium.hpp"


using namespace sm;
using namespace sm::literals;

#define IX(i, j) ((i) + (N + 2) * (j))

struct Fluid
{
    u64 N = 20;
    u64 size() const { return (N + 2) * (N + 2); }

    cuda::HostDevVec u         = cuda::HostDevVec(size());
    cuda::HostDevVec v         = cuda::HostDevVec(size());
    cuda::HostDevVec u_prev    = cuda::HostDevVec(size());
    cuda::HostDevVec dens      = cuda::HostDevVec(size());
    cuda::HostDevVec dens_prev = cuda::HostDevVec(size());
};

auto main() -> i32
{
    const auto N    = 20;
    const auto dims = Dimensions{N + 2, N + 2};
    auto window     = Window{{.dims{700, 700}}};
    window.display(); // to fix world_box?
    const auto viewport_box = window.world_box();

    auto watch         = Stopwatch{};
    auto frame_counter = 0;

    auto fluid = Fluid(N);

    // auto image = expect(file::read_image("D:\\sm\\india-map-2019.jpg"));
    auto image = Image{dims};

    auto texture = gl::Texture{gl::ImageFormat::RGBA8, dims, gl::Texture::Wrap::None,
                               gl::Texture::Filter::None, gl::Texture::Filter::Nearest};


    const auto draw_tex = [&]
    {
        texture.set_data(image, dims);
        using Vert                        = gl::Vertex<gl::Layout::PosTex>;
        static constexpr auto buffer_data = std::to_array<Vert>({{{-1, -1}, {0, 0}},
                                                                 {{1, 1}, {1, 1}},
                                                                 {{-1, 1}, {0, 1}},

                                                                 {{-1, -1}, {0, 0}},
                                                                 {{1, -1}, {1, 0}},
                                                                 {{1, 1}, {1, 1}}});

        auto& ctx          = window.context;
        const auto& shader = ctx.shaders.at("PosTex");
        ctx.set_active(shader);
        shader.set("view", glm::mat4{1.0F});

        texture.bind();

        auto& vao = ctx.vertex_arrays.at("PosTex");
        ctx.set_active(vao);

        const auto& buffer = ctx.vertex_buffers.at("default");
        buffer.set_data(buffer_data);
        vao.bind(buffer, sizeof(Vert));

        glDrawArrays(GL_TRIANGLES, 0, static_cast<i32>(buffer_data.size()));
    };

    const auto draw = [&]
    {
        // drawing mouse later so do bg last
        draw::background("#16161c"_c);

        for (const auto& [pos, _] : image.enumerate_2d())
        {
            const auto c =
                static_cast<u8>(static_cast<f32>((pos.x + frame_counter) % dims.x) / dims.x * 255);
            image[pos].r = c;
            image[pos].g = c;
            image[pos].b = c;
        }

        draw_tex();

        if (window.mouse.left)
        {
            draw::circle(window, {window.mouse_pos(), .08}, Color{132, 30, 252});
        }

        watch.reset();
        window.pan([&] { return window.mouse.middle; });
        window.zoom_to_cursor();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        frame_counter++;
    };

    run(window, draw);
}
