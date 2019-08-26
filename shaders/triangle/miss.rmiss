#version 460
#extension GL_NV_ray_tracing : require

layout(location = 0) rayPayloadInNV vec3 hitValue;

void main() {
    // Cornflower blue ftw
    hitValue = vec3(100.0/255.0, 149.0/255.0, 237.0/255.0);
}