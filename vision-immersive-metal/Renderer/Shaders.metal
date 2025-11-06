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
    
    float3 normalizedPos = normalize(in.position);
    float2 sphereCoords = float2(
        atan2(normalizedPos.z, normalizedPos.x), // theta [-pi, pi] -> will be remapped
        asin(normalizedPos.y)                    // phi [-pi/2, pi/2]
    );

    // Convert to UV space [0, 1] range
    float u = (sphereCoords.x + M_PI_F) / M_PI_F; // theta wraps around
    float v = (sphereCoords.y + (M_PI_2_F)) / M_PI_F; // phi clamped
    
    if (u > 0.998) {
        out.texCoord = float2(-1.0, -1.0);
    } else {
        out.texCoord = float2(u * 0.5 + amp_id * 0.5, 1.0 - v);
    }

//    out.texCoord = float2(in.texCoord.x * 0.5 + amp_id * 0.5, in.texCoord.y);

    return out;
}

[[fragment]]
float4 fragmentShader(ColorInOut in [[stage_in]],
                               texture2d<half> frameTexture [[ texture(TextureIndexColor) ]])
{
    if (in.texCoord.x == -1.0) {
        return float4(0, 0, 0, 1);
    }
//    constexpr sampler colorSampler(mip_filter::linear,
//                                   mag_filter::linear,
//                                   min_filter::linear);
    constexpr sampler bilinearSampler(address::clamp_to_edge, filter::linear, mip_filter::none);

    half4 colorSample   = frameTexture.sample(bilinearSampler, in.texCoord.xy);

    return float4(colorSample);
}
