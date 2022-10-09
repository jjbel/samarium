#include <string_view>

#include "gl.hpp"

#include "samarium/math/Extents.hpp"

namespace sm::gl
{
VertexArray::VertexArray() { glCreateVertexArrays(1, &handle); }

VertexArray::VertexArray(const std::vector<VertexAttribute>& attributes)
{
    glCreateVertexArrays(1, &handle);
    for (auto i : range(attributes.size())) { make_attribute(static_cast<u32>(i), attributes[i]); }
}

void VertexArray::bind()
{
    glBindVertexArray(handle); // make active, creating if necessary
}

void VertexArray::make_attribute(u32 index, const VertexAttribute& attribute)
{
    glEnableVertexArrayAttrib(handle, index);
    glVertexArrayAttribBinding(handle, index, 0);
    glVertexArrayAttribFormat(handle, index, attribute.size, attribute.type, attribute.normalized,
                              attribute.offset);
}

void VertexArray::bind(const VertexBuffer& buffer, i32 stride)
{
    glVertexArrayVertexBuffer(handle, 0, buffer.handle, 0, stride);
}

void VertexArray::bind(const ElementBuffer& buffer)
{
    glVertexArrayElementBuffer(handle, buffer.handle);
}

VertexArray::~VertexArray() { glDeleteVertexArrays(1, &handle); }
} // namespace sm::gl
