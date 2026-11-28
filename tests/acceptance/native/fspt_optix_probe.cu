// Bounded OptiX feasibility probe. This is not the FSPT rendering backend.
#include <optix.h>
#include <cuda_runtime.h>

struct ProbeParams {
    OptixTraversableHandle handle;
    float4* output;
};

extern "C" {
__constant__ ProbeParams params;
}

extern "C" __global__ void __raygen__probe() {
    const unsigned int index = optixGetLaunchIndex().x;
    const float3 origin = make_float3(index == 0 ? 0.0f : 2.0f, 0.0f, 1.0f);
    unsigned int t = __float_as_uint(-2.0f);
    unsigned int u = __float_as_uint(-2.0f);
    unsigned int v = __float_as_uint(-2.0f);
    unsigned int hit = 2;
    optixTrace(params.handle, origin, make_float3(0.0f, 0.0f, -1.0f),
               0.0f, 10.0f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0, 1, 0, t, u, v, hit);
    params.output[index] = make_float4(__uint_as_float(t), __uint_as_float(u),
                                       __uint_as_float(v), static_cast<float>(hit));
}

extern "C" __global__ void __miss__probe() {
    optixSetPayload_0(__float_as_uint(-1.0f));
    optixSetPayload_1(__float_as_uint(-1.0f));
    optixSetPayload_2(__float_as_uint(-1.0f));
    optixSetPayload_3(0);
}

extern "C" __global__ void __closesthit__probe() {
    const float2 barycentric = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(__float_as_uint(barycentric.x));
    optixSetPayload_2(__float_as_uint(barycentric.y));
    optixSetPayload_3(1);
}
