#include "range/v3/numeric/accumulate.hpp"
#include "range/v3/view/take.hpp"
#include "samarium/samarium.hpp"
#include "samarium/util/Stopwatch.hpp"

using namespace sm;
using namespace sm::literals;

constexpr auto src =
#include "samarium/physics/gpu/version.comp.glsl"

#include "samarium/physics/gpu/Particle.comp.glsl"

#include "samarium/physics/gpu/update.comp.glsl"
    ;

auto main() -> i32
{
    auto window = Window{{{1800, 900}}};
    auto buffer = expect(gl::MappedBuffer<Particle<f32>>::make(1024));
    buffer.fill(Particle<f32>{.pos{0, 0}, .vel{1, 2}});

    auto shader = expect(gl::ComputeShader::make(src));

    // https://juandiegomontoya.github.io/particles.html#gpu

    const auto update = [&]
    {
        const auto work_group_count = (buffer.data.size() + 64 - 1) / 64;
        shader.bind();
        buffer.bind(2);
        shader.run(static_cast<u32>(work_group_count));
        buffer.read();
        gl::sync();
    };

    auto watch = Stopwatch{};
    update();
    // update();

    watch.print();
    print("SUM:", ranges::accumulate(
                      buffer.data, Vector2f{}, [](Vector2f a, Vector2f b) { return a + b; },
                      &Particle<f32>::pos));
}
