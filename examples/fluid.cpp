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
    const u64 N;
    const f32 dt;
    const u64 size;
    const u64 byte_size;

    const f64 diffusion = 0.01F;
    const f64 visc      = 0.001F;

    cuda::HostDevVec u         = cuda::HostDevVec(size);
    cuda::HostDevVec v         = cuda::HostDevVec(size);
    cuda::HostDevVec u_prev    = cuda::HostDevVec(size);
    cuda::HostDevVec v_prev    = cuda::HostDevVec(size);
    cuda::HostDevVec dens      = cuda::HostDevVec(size);
    cuda::HostDevVec dens_prev = cuda::HostDevVec(size);
    // cuda::HostDevVec s         = cuda::HostDevVec(size);

    Fluid(u64 N = 20, f32 dt = 0.1F)
        : N{N}, dt{dt}, size{(N + 2) * (N + 2)}, byte_size{size * sizeof(f32)}
    {
        u.malloc_host();
        v.malloc_host();
        u_prev.malloc_host();
        v_prev.malloc_host();
        dens.malloc_host();
        dens_prev.malloc_host();
        // s.malloc_host();

        // swapping is done, so that prev is used both for taking user input (forces/sources)
        // and for calculation

        for (u64 i = 0; i < size; i++)
        {
            u.host[i]      = 0;
            v.host[i]      = 0;
            u_prev.host[i] = 0;
            v_prev.host[i] = 0;
            // s.host[i]      = 0;
        }
    }

    ~Fluid()
    {
        u.free_host();
        v.free_host();
        u_prev.free_host();
        v_prev.free_host();
        dens.free_host();
        dens_prev.free_host();
        // s.free_host();
    }

    void add_source(f32* dest, f32* source)
    {
        for (u64 i = 0; i < size; i++) { dest[i] += dt * source[i]; }
    }

    // x0 is before
    // x is after
    void diffuse(int b, float* x, float* x0, float diff)
    {
        const float a = dt * diff * N * N;

        for (u64 k = 0; k < 20; k++)
        {
            for (u64 i = 1; i <= N; i++)
            {
                for (u64 j = 1; j <= N; j++)
                {
                    x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                       x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                                  (1 + 4 * a);
                }
            }
            set_bnd(b, x);
        }
    }

    // (u, v) : velocity
    // d0: before
    // d: after
    void advect(int b, float* d, float* d0, float* u, float* v)
    {
        int i, j, i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1;
        const f32 dt0 = dt * N;
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                x = i - dt0 * u[IX(i, j)];
                y = j - dt0 * v[IX(i, j)];
                if (x < 0.5) x = 0.5;
                if (x > N + 0.5) x = N + 0.5;
                i0 = (int)x;
                i1 = i0 + 1;
                if (y < 0.5) y = 0.5;
                if (y > N + 0.5) y = N + 0.5;
                j0          = (int)y;
                j1          = j0 + 1;
                s1          = x - i0;
                s0          = 1 - s1;
                t1          = y - j0;
                t0          = 1 - t1;
                d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                              s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
            }
        }
        set_bnd(b, d);
    }

    void vel_step()
    {
        add_source(u.host, u_prev.host);
        add_source(v.host, v_prev.host);

        std::swap(u_prev.host, u.host);
        std::swap(v_prev.host, v.host);
        // diffuse(1, u.host, u_prev.host, visc);
        // diffuse(2, v.host, v_prev.host, visc);
        // project(N, u.host, v.host, u_prev.host, v_prev.host);
        std::swap(u_prev.host, u.host);
        std::swap(v_prev.host, v.host);

        // advect(1, u.host, u_prev.host, u_prev.host, v_prev.host);
        // advect(2, v.host, v_prev.host, u_prev.host, v_prev.host);
        // project(N, u.host, v.host, u_prev.host, v_prev.host);
    }

    void set_bnd(int b, f32* x)
    {
        // TODO swap the if and the for
        for (u64 i = 1; i <= N; i++)
        {
            x[IX(0, i)]     = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
            x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
            x[IX(i, 0)]     = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
            x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)];
        }

        x[IX(0, 0)]         = 0.5F * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, N + 1)]     = 0.5F * (x[IX(1, N + 1)] + x[IX(0, N)]);
        x[IX(N + 1, 0)]     = 0.5F * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
        x[IX(N + 1, N + 1)] = 0.5F * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    }

    void project(int N, float* u, float* v, float* p, float* div)
    {
        int i, j, k;
        float h;

        h = 1.0 / N;
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                div[IX(i, j)] =
                    -0.5 * h *
                    (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
                p[IX(i, j)] = 0;
            }
        }
        set_bnd(0, div);
        set_bnd(0, p);
        for (k = 0; k < 20; k++)
        {
            for (i = 1; i <= N; i++)
            {
                for (j = 1; j <= N; j++)
                {
                    p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                   p[IX(i, j - 1)] + p[IX(i, j + 1)]) /
                                  4;
                }
            }
            set_bnd(0, p);
        }
        for (i = 1; i <= N; i++)
        {
            for (j = 1; j <= N; j++)
            {
                u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
                v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
            }
        }
        set_bnd(1, u);
        set_bnd(2, v);
    }

    void dens_step()
    {
        // print(dens.host[23]);
        // add_source(dens.host, dens_prev.host);
        // print(dens_prev.host[23]);
        std::swap(dens.host, dens_prev.host);
        diffuse(0, dens.host, dens_prev.host, diffusion);
        std::swap(dens.host, dens_prev.host);
        // advect(0, dens.host, dens_prev.host, u.host, v.host);
    }

    void update()
    {
        // vel_step();
        dens_step();

        for (u64 i = 0; i < size; i++)
        {
            u_prev.host[i]    = 0;
            v_prev.host[i]    = 0;
            dens_prev.host[i] = 0;
        }
    }
};

auto main() -> i32
{
    const auto N    = 50;
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

    for (const auto& [pos, _] : image.enumerate_2d())
    {
        // fluid.dens.host[IX(pos.x, pos.y)] = std::sin(static_cast<f32>(pos.x + pos.y) / 2.0F +
        // 0.5F);
        const auto index       = IX(pos.x, pos.y);
        fluid.dens.host[index] = math::distance(pos.template cast<f64>(), Vec2{N / 2, N / 2}) < 10;
        fluid.dens_prev.host[index] =
            math::distance(pos.template cast<f64>(), Vec2{N / 2, N / 2}) < 10;

        if (10 <= pos.x && pos.x <= 25 && 23 <= pos.y && pos.y <= 25)
        {
            fluid.v.host[index] = (pos.x < N / 2) * 0.01F;
        }
    }

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

        fluid.update();

        for (const auto& [pos, _] : image.enumerate_2d())
        {
            if (1)
            {
                const auto d = fluid.dens.host[IX(pos.x, pos.y)];
                const auto c = static_cast<u8>(d * 255);
                image[pos].r = c;
                image[pos].g = c;
                image[pos].b = c;
            }
            else
            {
                const auto u = fluid.u.host[IX(pos.x, pos.y)];
                const auto v = fluid.v.host[IX(pos.x, pos.y)];
                const auto c = static_cast<u8>(255 / (math::pi / 2) *
                                               std::atan(std::sqrt(u * u + v * v) * 100));
                image[pos].r = static_cast<u8>(255 * 100 * std::abs(u));
                image[pos].g = 0;
                image[pos].b = static_cast<u8>(255 * 100 * std::abs(v));
            }
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
