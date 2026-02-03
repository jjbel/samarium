#include "samarium/samarium.hpp"

using namespace sm;

void print_heights(const std::vector<f32>& heights)
{
    fmt::print("[");
    for (u64 i = 0; i < heights.size(); i++) { fmt::print("{:5.2f} ", heights[i]); }
    fmt::print("]\n");
}

auto main() -> i32
{
    auto window = Window{{.dims = {1280, 720}}};

    const u64 Nx = 50;
    const u64 Nt = 100;
    const f32 dx = 0.1;
    const f32 dt = 0.1;

    const f64 c = 1.0;

    // Initial conditions
    // const auto initial = std::vector<f32>{0, 0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, 0, 0, 0};

    // -----------------------------
    auto u = std::vector<std::vector<f32>>(Nt + 1, std::vector<f32>(Nx + 1));

    auto n          = 1;
    const auto C    = c * dt / dx; // Courant Number
    const auto C_sq = C * C;

    // initial conditions:
    // u[0] = initial;
    u[0][20] = 0.3f;
    u[0][21] = 0.3f;

    // finding u[1]
    for (u64 i = 1; i <= Nx - 1; i++)
    {
        u[1][i] = u[0][i] - C_sq / 2 * (u[0][i + 1] - 2 * u[0][i] + u[0][i - 1]);
    }

    const auto update = [&]
    {
        if (n == Nt - 1) { return; }
        for (u64 i = 1; i <= Nx; i++)
        {
            u[n + 1][i] =
                -u[n - 1][i] + 2 * u[n][i] + C_sq * (u[n][i + 1] - 2 * u[n][i] + u[n][i - 1]);
        }

        // Boundary conditions:
        u[n + 1][0]  = 0;
        u[n + 1][Nx] = 0;

        fmt::print("{:4} ", n);

        n++;
    };

    const auto make_points = [](const auto& heights)
    {
        auto vec = std::vector<Vec2f>(heights.size());
        for (u64 i = 0; i < heights.size(); i++)
        {
            vec[i].x = interp::map_range<f32>(i, {0.0f, f32(heights.size())}, {-1.0f, 1.0f});
            vec[i].y = heights[i];
        }
        return vec;
    };

    u64 frame = 0;

    const auto draw = [&]
    {
        window.zoom_to_cursor();
        window.pan();

        draw::background(Color{});
        print_heights(u[n]);

        const auto pts = make_points(u[n]);
        draw::polyline_segments(window, pts, 0.01f, Color{255, 255, 255});
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        update();
        // print(frame);
        frame++;
    };

    run(window, draw);
}
