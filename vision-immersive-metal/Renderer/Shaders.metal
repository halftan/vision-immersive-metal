//
//  Shaders.metal
//  vision-immersive-metal
//
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

[[vertex]]
ColorInOut vertexShader(Vertex in [[stage_in]],
                               ushort amp_id [[amplification_id]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               constant ViewProjectionArray & viewProjectionArray [[ buffer(BufferIndexViewProjection) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = viewProjectionArray.viewProjectionMatrix[amp_id] * uniforms.modelMatrix * position;

    // Use precomputed texture coordinates from the mesh
    // The mesh already generates proper equirectangular UV coordinates
    out.texCoord = in.texCoord.xy;

    return out;
}

[[fragment]]
float4 fragmentShader(ColorInOut in [[stage_in]],
                      ushort amp_id [[amplification_id]],
                      texture2d<half> frameTexture [[ texture(TextureIndexColor) ]])
{
    constexpr sampler bilinearSampler(address::clamp_to_edge, filter::linear, mip_filter::none);

    // Apply stereo sampling for VR180 side-by-side content
    // Hemisphere UV coordinates are already 0-0.5 range
    float2 sampleCoord = float2(in.texCoord.x + amp_id * 0.5, 1.0 - in.texCoord.y);

    half4 colorSample = frameTexture.sample(bilinearSampler, sampleCoord);

    return float4(colorSample);
}
