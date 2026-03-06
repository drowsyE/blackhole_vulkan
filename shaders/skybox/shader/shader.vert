#version 450

layout(location = 0) out vec3 outDir;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} ubo;

// y flip needed in view matrix!! (Vulkan -y가 위 -> OpenGL +y가 위)
vec3 positions[36] = vec3[](
    // +X
    vec3( 1.0, -1.0, -1.0),
    vec3( 1.0, -1.0,  1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3( 1.0,  1.0, -1.0),

    // -X
    vec3(-1.0, -1.0,  1.0),
    vec3(-1.0, -1.0, -1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3(-1.0, -1.0,  1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3(-1.0,  1.0,  1.0),

    // +Y
    vec3(-1.0,  1.0, -1.0),
    vec3( 1.0,  1.0, -1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3(-1.0,  1.0,  1.0),

    // -Y
    vec3(-1.0, -1.0,  1.0),
    vec3( 1.0, -1.0,  1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3(-1.0, -1.0,  1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3(-1.0, -1.0, -1.0),

    // +Z
    vec3(-1.0, -1.0,  1.0),
    vec3(-1.0,  1.0,  1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3(-1.0, -1.0,  1.0),
    vec3( 1.0,  1.0,  1.0),
    vec3( 1.0, -1.0,  1.0),

    // -Z
    vec3( 1.0, -1.0, -1.0),
    vec3( 1.0,  1.0, -1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3( 1.0, -1.0, -1.0),
    vec3(-1.0,  1.0, -1.0),
    vec3(-1.0, -1.0, -1.0)
);


void main() {
    vec3 pos = positions[gl_VertexIndex];
    outDir = pos;

    mat4 viewNoTranslation = mat4(mat3(ubo.view));
    vec4 clipPos = ubo.proj * viewNoTranslation * vec4(pos, 1.0);
    gl_Position = clipPos.xyww;
}
