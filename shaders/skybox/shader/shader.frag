#version 450

layout(set = 1, binding = 0) uniform samplerCube skybox;

layout(location = 0) in vec3 inDir;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 dir = inDir;
    dir.z *= -1;
    outColor = texture(skybox, dir);
}