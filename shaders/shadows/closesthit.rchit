#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV vec3 hitValue;
layout(location = 1) rayPayloadNV bool shadowed;

hitAttributeNV vec3 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureNV topLevelAS;
layout(binding = 3, set = 0) buffer Vertices { vec4 v[]; } vertices;
layout(binding = 4, set = 0) buffer Indices { uint i[]; } indices;

struct Vertex {
  vec3 position;
  vec3 normal;
};

Vertex unpack(uint index) {
	vec4 d0 = vertices.v[2 * index + 0];
	vec4 d1 = vertices.v[2 * index + 1];

	Vertex v;
	v.position = d0.xyz;
	v.normal = d1.xyz;
	return v;
}

const vec3 LIGHT_DIR = normalize(vec3(0.5, 1.0, -0.5));
const uint CULL_MASK = 0xff;
const float T_MIN = 0.01;
const float T_MAX = 10.0;

void main() {
	ivec3 index = ivec3(indices.i[3 * gl_PrimitiveID], indices.i[3 * gl_PrimitiveID + 1], indices.i[3 * gl_PrimitiveID + 2]);

	Vertex v0 = unpack(index.x);
	Vertex v1 = unpack(index.y);
	Vertex v2 = unpack(index.z);

  	// Interpolate and transform normal
	vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec4 ogNormal = vec4(normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z), 0.0);
	vec3 normal = normalize((gl_ObjectToWorldNV * ogNormal).xyz);

  	// Basic lighting
	hitValue = vec3(max(dot(LIGHT_DIR, normal), 0.0)) * 0.8;

	shadowed = true;

	// Cast new ray in light direction
	vec3 origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;

	traceNV(
		topLevelAS, 
		gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsOpaqueNV | gl_RayFlagsSkipClosestHitShaderNV, 
		CULL_MASK, 
		1, 0, 1, 
		origin, 
		T_MIN, 
		LIGHT_DIR, 
		T_MAX, 
		1);

	if (shadowed) {
		hitValue *= 0.3;
	}
}