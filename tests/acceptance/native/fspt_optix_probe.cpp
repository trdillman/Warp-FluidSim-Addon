// Bounded C ABI probe of OptiX on the existing Warp CUDA context.
// No CPython headers, Python extension ABI, runtime replacement, or GPU display.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define FSPT_EXPORT extern "C" __declspec(dllexport)
#else
#define FSPT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct ProbeStats {
    std::uint32_t abi_version;
    std::uint32_t optix_version;
    std::uint64_t context;
    std::uint64_t stream;
    std::uint64_t gas_bytes;
    std::uint64_t scratch_bytes;
    double wall_ms;
    std::int32_t retirement_result;
    std::int32_t cleanup_result;
};

struct ProbeParams {
    OptixTraversableHandle handle;
    std::uint64_t output;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

static void cuda_check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " CUDA result " + std::to_string(result));
    }
}

static void optix_check(OptixResult result, const char* operation) {
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " OptiX result " + std::to_string(result));
    }
}

struct Resources {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup groups[3] = {};
    std::vector<CUdeviceptr> allocations;
    // If retirement fails, this owner is intentionally retained until the
    // disposable probe process exits. No uncertain in-flight storage is freed.
    int cleanup() {
        int first = 0;
        auto save = [&first](int result) { if (first == 0 && result != 0) first = result; };
        for (auto ptr : allocations) save(static_cast<int>(cuMemFree(ptr)));
        allocations.clear();
        if (pipeline) save(static_cast<int>(optixPipelineDestroy(pipeline)));
        for (auto group : groups) if (group) save(static_cast<int>(optixProgramGroupDestroy(group)));
        if (module) save(static_cast<int>(optixModuleDestroy(module)));
        if (context) save(static_cast<int>(optixDeviceContextDestroy(context)));
        return first;
    }
    CUdeviceptr allocate(std::size_t bytes) {
        CUdeviceptr pointer = 0;
        cuda_check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
        allocations.push_back(pointer);
        return pointer;
    }
};

static std::vector<Resources*> quarantined;

FSPT_EXPORT std::uint32_t fspt_optix_probe_abi() { return 1; }

FSPT_EXPORT int fspt_optix_triangle(
    const char* ptx, std::size_t ptx_size, void* expected_context, void* raw_stream,
    std::uint64_t vertices, std::uint64_t indices, std::uint64_t output,
    char* diagnostic, std::size_t diagnostic_size, ProbeStats* stats) {
    if (!stats || !diagnostic || diagnostic_size < 2) return 2;
    *stats = {};
    stats->abi_version = 1;
    stats->optix_version = OPTIX_VERSION;
    stats->retirement_result = -1;
    diagnostic[0] = '\0';
    const auto start = std::chrono::steady_clock::now();
    Resources* owned = nullptr;
    CUstream stream = reinterpret_cast<CUstream>(raw_stream);
    bool may_have_work = false;
    try {
        if (!quarantined.empty()) throw std::runtime_error("previous probe retirement is quarantined");
        if (!ptx || ptx_size == 0 || !raw_stream || !vertices || !indices || !output) {
            throw std::runtime_error("missing probe inputs");
        }
        CUcontext current = nullptr;
        CUcontext stream_context = nullptr;
        cuda_check(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (!current || current != expected_context) throw std::runtime_error("CUDA context mismatch");
        cuda_check(cuStreamGetCtx(stream, &stream_context), "cuStreamGetCtx");
        if (stream_context != current) throw std::runtime_error("CUDA stream context mismatch");
        stats->context = reinterpret_cast<std::uint64_t>(current);
        stats->stream = reinterpret_cast<std::uint64_t>(stream);
        optix_check(optixInit(), "optixInit");
        owned = new Resources();
        OptixDeviceContextOptions context_options = {};
        context_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        optix_check(optixDeviceContextCreate(current, &context_options, &owned->context),
                    "optixDeviceContextCreate");

        const std::uint32_t flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
        CUdeviceptr vertex_pointer = vertices;
        OptixBuildInput geometry = {};
        geometry.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        geometry.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        geometry.triangleArray.vertexStrideInBytes = 12;
        geometry.triangleArray.numVertices = 3;
        geometry.triangleArray.vertexBuffers = &vertex_pointer;
        geometry.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        geometry.triangleArray.indexStrideInBytes = 12;
        geometry.triangleArray.numIndexTriplets = 1;
        geometry.triangleArray.indexBuffer = indices;
        geometry.triangleArray.flags = flags;
        geometry.triangleArray.numSbtRecords = 1;
        OptixAccelBuildOptions build = {};
        build.buildFlags = OPTIX_BUILD_FLAG_NONE;
        build.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes = {};
        optix_check(optixAccelComputeMemoryUsage(owned->context, &build, &geometry, 1, &sizes),
                    "optixAccelComputeMemoryUsage");
        stats->gas_bytes = sizes.outputSizeInBytes;
        stats->scratch_bytes = sizes.tempSizeInBytes;
        const CUdeviceptr scratch = owned->allocate(sizes.tempSizeInBytes);
        const CUdeviceptr gas = owned->allocate(sizes.outputSizeInBytes);
        OptixTraversableHandle handle = 0;
        may_have_work = true;
        optix_check(optixAccelBuild(owned->context, stream, &build, &geometry, 1,
                                   scratch, sizes.tempSizeInBytes, gas, sizes.outputSizeInBytes,
                                   &handle, nullptr, 0), "optixAccelBuild");

        OptixModuleCompileOptions module_options = {};
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        OptixPipelineCompileOptions compile = {};
        compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile.numPayloadValues = 4;
        compile.numAttributeValues = 2;
        compile.pipelineLaunchParamsVariableName = "params";
        compile.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        char log[4096] = {};
        std::size_t log_size = sizeof(log);
        const auto module_result = optixModuleCreate(owned->context, &module_options, &compile,
                                                     ptx, ptx_size, log, &log_size, &owned->module);
        if (module_result != OPTIX_SUCCESS) {
            throw std::runtime_error(std::string("optixModuleCreate: ") + log);
        }
        OptixProgramGroupOptions group_options = {};
        OptixProgramGroupDesc descriptions[3] = {};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen.module = owned->module;
        descriptions[0].raygen.entryFunctionName = "__raygen__probe";
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss.module = owned->module;
        descriptions[1].miss.entryFunctionName = "__miss__probe";
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleCH = owned->module;
        descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__probe";
        for (int index = 0; index < 3; ++index) {
            log_size = sizeof(log);
            optix_check(optixProgramGroupCreate(owned->context, &descriptions[index], 1,
                        &group_options, log, &log_size, &owned->groups[index]), "optixProgramGroupCreate");
        }
        OptixPipelineLinkOptions link = {};
        link.maxTraceDepth = 1;
        log_size = sizeof(log);
        optix_check(optixPipelineCreate(owned->context, &compile, &link, owned->groups, 3,
                                        log, &log_size, &owned->pipeline), "optixPipelineCreate");
        OptixStackSizes stack = {};
        for (auto group : owned->groups) {
            optix_check(optixUtilAccumulateStackSizes(group, &stack, owned->pipeline),
                        "optixUtilAccumulateStackSizes");
        }
        std::uint32_t traversal = 0, state = 0, continuation = 0;
        optix_check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation),
                    "optixUtilComputeStackSizes");
        optix_check(optixPipelineSetStackSize(owned->pipeline, traversal, state, continuation, 1),
                    "optixPipelineSetStackSize");

        Record records[3] = {};
        for (int index = 0; index < 3; ++index) {
            optix_check(optixSbtRecordPackHeader(owned->groups[index], &records[index]),
                        "optixSbtRecordPackHeader");
        }
        const CUdeviceptr device_records = owned->allocate(sizeof(records));
        // Tiny pageable control records use a completed host copy. This probe
        // deliberately makes no asynchronous preparation/performance claim.
        cuda_check(cuMemcpyHtoD(device_records, records, sizeof(records)), "SBT upload");
        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = device_records;
        sbt.missRecordBase = device_records + sizeof(Record);
        sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = device_records + 2 * sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = 1;
        const ProbeParams parameters = {handle, output};
        const CUdeviceptr device_parameters = owned->allocate(sizeof(parameters));
        cuda_check(cuMemcpyHtoD(device_parameters, &parameters, sizeof(parameters)),
                    "launch parameters upload");
        optix_check(optixLaunch(owned->pipeline, stream, device_parameters, sizeof(parameters),
                                &sbt, 2, 1, 1), "optixLaunch");
        stats->retirement_result = static_cast<int>(cuStreamSynchronize(stream));
        cuda_check(static_cast<CUresult>(stats->retirement_result), "probe stream retirement");
        may_have_work = false;
        stats->cleanup_result = owned->cleanup();
        delete owned;
        owned = nullptr;
        if (stats->cleanup_result != 0) throw std::runtime_error("probe cleanup failed");
        stats->wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::snprintf(diagnostic, diagnostic_size, "OptiX GAS and two rays completed on supplied CUDA context");
        return 0;
    } catch (const std::exception& error) {
        if (owned) {
            if (may_have_work) stats->retirement_result = static_cast<int>(cuStreamSynchronize(stream));
            if (!may_have_work || stats->retirement_result == 0) {
                stats->cleanup_result = owned->cleanup();
                delete owned;
            } else {
                quarantined.push_back(owned);
            }
        }
        stats->wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::snprintf(diagnostic, diagnostic_size, "%s", error.what());
        return 1;
    }
}
