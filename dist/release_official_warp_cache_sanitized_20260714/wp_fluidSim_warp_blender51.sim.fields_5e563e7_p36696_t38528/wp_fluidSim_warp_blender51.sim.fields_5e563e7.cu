
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:34
static CUDA_CALLABLE wp::int32 leaf_linear_index_0(
    wp::vec_t<3, wp::int32> var_lc,
    wp::vec_t<3, wp::int32> var_leaves_dim)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    //---------
    // forward
    // def leaf_linear_index(lc: wp.vec3i, leaves_dim: wp.vec3i) -> wp.int32:                 <L 35>
    // return lc[0] + leaves_dim[0] * (lc[1] + leaves_dim[1] * lc[2])                         <L 37>
    var_1 = wp::extract(var_lc, var_0);
    var_3 = wp::extract(var_leaves_dim, var_2);
    var_5 = wp::extract(var_lc, var_4);
    var_7 = wp::extract(var_leaves_dim, var_6);
    var_9 = wp::extract(var_lc, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_11 = wp::add(var_5, var_10);
    var_12 = wp::mul(var_3, var_11);
    var_13 = wp::add(var_1, var_12);
    return var_13;
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:112
static CUDA_CALLABLE wp::vec_t<3, wp::float32> world_to_voxel_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_origin,
    wp::float32 var_voxel_size)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 1;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    //---------
    // forward
    // def world_to_voxel(p: wp.vec3, origin: wp.vec3, voxel_size: float) -> wp.vec3:         <L 113>
    // inv = 1.0 / voxel_size                                                                 <L 115>
    var_1 = wp::div(var_0, var_voxel_size);
    // return wp.vec3((p[0] - origin[0]) * inv, (p[1] - origin[1]) * inv, (p[2] - origin[2]) * inv)       <L 116>
    var_3 = wp::extract(var_p, var_2);
    var_5 = wp::extract(var_origin, var_4);
    var_6 = wp::sub(var_3, var_5);
    var_7 = wp::mul(var_6, var_1);
    var_9 = wp::extract(var_p, var_8);
    var_11 = wp::extract(var_origin, var_10);
    var_12 = wp::sub(var_9, var_11);
    var_13 = wp::mul(var_12, var_1);
    var_15 = wp::extract(var_p, var_14);
    var_17 = wp::extract(var_origin, var_16);
    var_18 = wp::sub(var_15, var_17);
    var_19 = wp::mul(var_18, var_1);
    var_20 = wp::vec_t<3, wp::float32>(var_7, var_13, var_19);
    return var_20;
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:119
static CUDA_CALLABLE wp::vec_t<3, wp::int32> voxel_floor_0(
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::int32 var_11;
    wp::vec_t<3, wp::int32> var_12;
    //---------
    // forward
    // def voxel_floor(p: wp.vec3) -> wp.vec3i:                                               <L 120>
    // return wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))         <L 121>
    var_1 = wp::extract(var_p, var_0);
    var_2 = wp::floor(var_1);
    var_3 = wp::int(var_2);
    var_5 = wp::extract(var_p, var_4);
    var_6 = wp::floor(var_5);
    var_7 = wp::int(var_6);
    var_9 = wp::extract(var_p, var_8);
    var_10 = wp::floor(var_9);
    var_11 = wp::int(var_10);
    var_12 = wp::vec_t<3, wp::int32>(var_3, var_7, var_11);
    return var_12;
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:55
static CUDA_CALLABLE bool voxel_in_domain_0(
    wp::vec_t<3, wp::int32> var_ijk,
    wp::vec_t<3, wp::int32> var_voxels_dim)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    bool var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 0;
    bool var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    bool var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    const wp::int32 var_19 = 1;
    wp::int32 var_20;
    bool var_21;
    const wp::int32 var_22 = 2;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    bool var_26;
    bool var_27;
    //---------
    // forward
    // def voxel_in_domain(ijk: wp.vec3i, voxels_dim: wp.vec3i) -> wp.bool:                   <L 56>
    // return (                                                                               <L 57>
    // ijk[0] >= 0                                                                            <L 58>
    var_1 = wp::extract(var_ijk, var_0);
    var_3 = (var_1 >= var_2);
    // and ijk[1] >= 0                                                                        <L 59>
    var_5 = wp::extract(var_ijk, var_4);
    var_7 = (var_5 >= var_6);
    // and ijk[2] >= 0                                                                        <L 60>
    var_9 = wp::extract(var_ijk, var_8);
    var_11 = (var_9 >= var_10);
    // and ijk[0] < voxels_dim[0]                                                             <L 61>
    var_13 = wp::extract(var_ijk, var_12);
    var_15 = wp::extract(var_voxels_dim, var_14);
    var_16 = (var_13 < var_15);
    // and ijk[1] < voxels_dim[1]                                                             <L 62>
    var_18 = wp::extract(var_ijk, var_17);
    var_20 = wp::extract(var_voxels_dim, var_19);
    var_21 = (var_18 < var_20);
    // and ijk[2] < voxels_dim[2]                                                             <L 63>
    var_23 = wp::extract(var_ijk, var_22);
    var_25 = wp::extract(var_voxels_dim, var_24);
    var_26 = (var_23 < var_25);
    var_27 = var_3 && var_7 && var_11 && var_16 && var_21 && var_26;
    return var_27;
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:40
static CUDA_CALLABLE wp::vec_t<3, wp::int32> leaf_coord_of_voxel_0(
    wp::vec_t<3, wp::int32> var_ijk)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 3;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::int32 var_6;
    const wp::int32 var_7 = 2;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::int32> var_10;
    //---------
    // forward
    // def leaf_coord_of_voxel(ijk: wp.vec3i) -> wp.vec3i:                                    <L 41>
    // return wp.vec3i(ijk[0] >> LEAF_SHIFT, ijk[1] >> LEAF_SHIFT, ijk[2] >> LEAF_SHIFT)       <L 42>
    var_1 = wp::extract(var_ijk, var_0);
    var_3 = wp::rshift(var_1, var_2);
    var_5 = wp::extract(var_ijk, var_4);
    var_6 = wp::rshift(var_5, var_2);
    var_8 = wp::extract(var_ijk, var_7);
    var_9 = wp::rshift(var_8, var_2);
    var_10 = wp::vec_t<3, wp::int32>(var_3, var_6, var_9);
    return var_10;
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:34
static CUDA_CALLABLE void adj_leaf_linear_index_0(
    wp::vec_t<3, wp::int32> var_lc,
    wp::vec_t<3, wp::int32> var_leaves_dim,
    wp::vec_t<3, wp::int32> & adj_lc,
    wp::vec_t<3, wp::int32> & adj_leaves_dim,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:112
static CUDA_CALLABLE void adj_world_to_voxel_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_origin,
    wp::float32 var_voxel_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_origin,
    wp::float32 & adj_voxel_size,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:119
static CUDA_CALLABLE void adj_voxel_floor_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:55
static CUDA_CALLABLE void adj_voxel_in_domain_0(
    wp::vec_t<3, wp::int32> var_ijk,
    wp::vec_t<3, wp::int32> var_voxels_dim,
    wp::vec_t<3, wp::int32> & adj_ijk,
    wp::vec_t<3, wp::int32> & adj_voxels_dim,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_native_interop\dist\release_official_profile_sanitized_20260714\scripts\addons\fluidSim_warp_blender51\sim\fields.py:40
static CUDA_CALLABLE void adj_leaf_coord_of_voxel_0(
    wp::vec_t<3, wp::int32> var_ijk,
    wp::vec_t<3, wp::int32> & adj_ijk,
    wp::vec_t<3, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void k_mark_leaves_box_2018f68d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::vec_t<3, wp::int32> var_lo,
    wp::vec_t<3, wp::int32> var_hi,
    wp::vec_t<3, wp::int32> var_leaves_dim,
    wp::array_t<wp::int32> var_leaf_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32 var_2;
        const wp::int32 var_3 = 3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 2;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::int32> var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 2;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::int32> var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        const wp::int32 var_36 = 2;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        bool var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 2;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::vec_t<3, wp::int32> var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::int32 var_67 = 0;
        bool var_68;
        const wp::int32 var_69 = 2;
        wp::int32 var_70;
        const wp::int32 var_71 = 0;
        bool var_72;
        bool var_73;
        const wp::int32 var_74 = 0;
        wp::int32 var_75;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        bool var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 1;
        wp::int32 var_82;
        bool var_83;
        const wp::int32 var_84 = 2;
        wp::int32 var_85;
        const wp::int32 var_86 = 2;
        wp::int32 var_87;
        bool var_88;
        bool var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        //---------
        // forward
        // def k_mark_leaves_box(                                                                 <L 297>
        // tid = wp.tid()                                                                         <L 304>
        var_0 = builtin_tid1d();
        // llo = wp.vec3i(lo[0] >> LEAF_SHIFT, lo[1] >> LEAF_SHIFT, lo[2] >> LEAF_SHIFT)          <L 305>
        var_2 = wp::extract(var_lo, var_1);
        var_4 = wp::rshift(var_2, var_3);
        var_6 = wp::extract(var_lo, var_5);
        var_7 = wp::rshift(var_6, var_3);
        var_9 = wp::extract(var_lo, var_8);
        var_10 = wp::rshift(var_9, var_3);
        var_11 = wp::vec_t<3, wp::int32>(var_4, var_7, var_10);
        // lhi = wp.vec3i(hi[0] >> LEAF_SHIFT, hi[1] >> LEAF_SHIFT, hi[2] >> LEAF_SHIFT)          <L 306>
        var_13 = wp::extract(var_hi, var_12);
        var_14 = wp::rshift(var_13, var_3);
        var_16 = wp::extract(var_hi, var_15);
        var_17 = wp::rshift(var_16, var_3);
        var_19 = wp::extract(var_hi, var_18);
        var_20 = wp::rshift(var_19, var_3);
        var_21 = wp::vec_t<3, wp::int32>(var_14, var_17, var_20);
        // nx = lhi[0] - llo[0] + 1                                                               <L 307>
        var_23 = wp::extract(var_21, var_22);
        var_25 = wp::extract(var_11, var_24);
        var_26 = wp::sub(var_23, var_25);
        var_28 = wp::add(var_26, var_27);
        // ny = lhi[1] - llo[1] + 1                                                               <L 308>
        var_30 = wp::extract(var_21, var_29);
        var_32 = wp::extract(var_11, var_31);
        var_33 = wp::sub(var_30, var_32);
        var_35 = wp::add(var_33, var_34);
        // nz = lhi[2] - llo[2] + 1                                                               <L 309>
        var_37 = wp::extract(var_21, var_36);
        var_39 = wp::extract(var_11, var_38);
        var_40 = wp::sub(var_37, var_39);
        var_42 = wp::add(var_40, var_41);
        // if tid >= nx * ny * nz:                                                                <L 310>
        var_43 = wp::mul(var_28, var_35);
        var_44 = wp::mul(var_43, var_42);
        var_45 = (var_0 >= var_44);
        if (var_45) {
            // return                                                                             <L 311>
            continue;
        }
        // x = tid % nx                                                                           <L 312>
        var_46 = wp::mod(var_0, var_28);
        // y = (tid // nx) % ny                                                                   <L 313>
        var_47 = wp::floordiv(var_0, var_28);
        var_48 = wp::mod(var_47, var_35);
        // z = tid // (nx * ny)                                                                   <L 314>
        var_49 = wp::mul(var_28, var_35);
        var_50 = wp::floordiv(var_0, var_49);
        // lc = wp.vec3i(llo[0] + x, llo[1] + y, llo[2] + z)                                      <L 315>
        var_52 = wp::extract(var_11, var_51);
        var_53 = wp::add(var_52, var_46);
        var_55 = wp::extract(var_11, var_54);
        var_56 = wp::add(var_55, var_48);
        var_58 = wp::extract(var_11, var_57);
        var_59 = wp::add(var_58, var_50);
        var_60 = wp::vec_t<3, wp::int32>(var_53, var_56, var_59);
        // if lc[0] < 0 or lc[1] < 0 or lc[2] < 0:                                                <L 316>
        var_62 = wp::extract(var_60, var_61);
        var_64 = (var_62 < var_63);
        var_66 = wp::extract(var_60, var_65);
        var_68 = (var_66 < var_67);
        var_70 = wp::extract(var_60, var_69);
        var_72 = (var_70 < var_71);
        var_73 = var_64 || var_68 || var_72;
        if (var_73) {
            // return                                                                             <L 317>
            continue;
        }
        // if lc[0] >= leaves_dim[0] or lc[1] >= leaves_dim[1] or lc[2] >= leaves_dim[2]:         <L 318>
        var_75 = wp::extract(var_60, var_74);
        var_77 = wp::extract(var_leaves_dim, var_76);
        var_78 = (var_75 >= var_77);
        var_80 = wp::extract(var_60, var_79);
        var_82 = wp::extract(var_leaves_dim, var_81);
        var_83 = (var_80 >= var_82);
        var_85 = wp::extract(var_60, var_84);
        var_87 = wp::extract(var_leaves_dim, var_86);
        var_88 = (var_85 >= var_87);
        var_89 = var_78 || var_83 || var_88;
        if (var_89) {
            // return                                                                             <L 319>
            continue;
        }
        // leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1                                      <L 320>
        var_91 = leaf_linear_index_0(var_60, var_leaves_dim);
        wp::array_store(var_leaf_flags, var_91, var_90);
    }
}



extern "C" __global__ void k_expand_leaf_flags_b70a6ce9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_leaf_flags_in,
    wp::vec_t<3, wp::int32> var_leaves_dim,
    wp::array_t<wp::int32> var_leaf_flags_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32 var_2;
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        const wp::int32 var_12 = -1;
        const wp::int32 var_13 = -1;
        const wp::int32 var_14 = -1;
        const wp::int32 var_15 = 2;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        bool var_27;
        bool var_28;
        bool var_29;
        const wp::int32 var_30 = 2;
        wp::int32 var_31;
        bool var_32;
        bool var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        const wp::int32 var_45 = -1;
        const wp::int32 var_46 = 2;
        wp::range_t var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        const wp::int32 var_56 = 0;
        bool var_57;
        bool var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        bool var_62;
        bool var_63;
        const wp::int32 var_64 = 2;
        wp::int32 var_65;
        bool var_66;
        bool var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32* var_75;
        const wp::int32 var_76 = 0;
        bool var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 1;
        const wp::int32 var_82 = -1;
        const wp::int32 var_83 = 2;
        wp::range_t var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        const wp::int32 var_89 = 0;
        bool var_90;
        const wp::int32 var_91 = 0;
        bool var_92;
        const wp::int32 var_93 = 0;
        bool var_94;
        bool var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::int32 var_98;
        bool var_99;
        bool var_100;
        const wp::int32 var_101 = 2;
        wp::int32 var_102;
        bool var_103;
        bool var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        const wp::int32 var_113 = 0;
        bool var_114;
        wp::int32 var_115;
        const wp::int32 var_116 = 1;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        const wp::int32 var_119 = -1;
        const wp::int32 var_120 = -1;
        const wp::int32 var_121 = 2;
        wp::range_t var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 0;
        bool var_128;
        const wp::int32 var_129 = 0;
        bool var_130;
        const wp::int32 var_131 = 0;
        bool var_132;
        bool var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        bool var_137;
        bool var_138;
        const wp::int32 var_139 = 2;
        wp::int32 var_140;
        bool var_141;
        bool var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        wp::int32 var_145;
        wp::int32 var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::int32* var_150;
        const wp::int32 var_151 = 0;
        bool var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 1;
        wp::int32 var_155;
        const wp::int32 var_156 = 0;
        const wp::int32 var_157 = -1;
        const wp::int32 var_158 = 2;
        wp::range_t var_159;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        const wp::int32 var_164 = 0;
        bool var_165;
        const wp::int32 var_166 = 0;
        bool var_167;
        const wp::int32 var_168 = 0;
        bool var_169;
        bool var_170;
        wp::int32 var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        bool var_174;
        bool var_175;
        const wp::int32 var_176 = 2;
        wp::int32 var_177;
        bool var_178;
        bool var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::int32 var_185;
        wp::int32 var_186;
        wp::int32* var_187;
        const wp::int32 var_188 = 0;
        bool var_189;
        wp::int32 var_190;
        const wp::int32 var_191 = 1;
        wp::int32 var_192;
        const wp::int32 var_193 = 1;
        const wp::int32 var_194 = -1;
        const wp::int32 var_195 = 2;
        wp::range_t var_196;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        const wp::int32 var_201 = 0;
        bool var_202;
        const wp::int32 var_203 = 0;
        bool var_204;
        const wp::int32 var_205 = 0;
        bool var_206;
        bool var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        bool var_211;
        bool var_212;
        const wp::int32 var_213 = 2;
        wp::int32 var_214;
        bool var_215;
        bool var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        wp::int32 var_220;
        wp::int32 var_221;
        wp::int32 var_222;
        wp::int32 var_223;
        wp::int32* var_224;
        const wp::int32 var_225 = 0;
        bool var_226;
        wp::int32 var_227;
        const wp::int32 var_228 = 1;
        wp::int32 var_229;
        const wp::int32 var_230 = 1;
        const wp::int32 var_231 = -1;
        const wp::int32 var_232 = -1;
        const wp::int32 var_233 = 2;
        wp::range_t var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        const wp::int32 var_239 = 0;
        bool var_240;
        const wp::int32 var_241 = 0;
        bool var_242;
        const wp::int32 var_243 = 0;
        bool var_244;
        bool var_245;
        wp::int32 var_246;
        wp::int32 var_247;
        wp::int32 var_248;
        bool var_249;
        bool var_250;
        const wp::int32 var_251 = 2;
        wp::int32 var_252;
        bool var_253;
        bool var_254;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        wp::int32 var_258;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::int32* var_262;
        const wp::int32 var_263 = 0;
        bool var_264;
        wp::int32 var_265;
        const wp::int32 var_266 = 1;
        wp::int32 var_267;
        const wp::int32 var_268 = 0;
        const wp::int32 var_269 = -1;
        const wp::int32 var_270 = 2;
        wp::range_t var_271;
        wp::int32 var_272;
        wp::int32 var_273;
        wp::int32 var_274;
        wp::int32 var_275;
        const wp::int32 var_276 = 0;
        bool var_277;
        const wp::int32 var_278 = 0;
        bool var_279;
        const wp::int32 var_280 = 0;
        bool var_281;
        bool var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        bool var_286;
        bool var_287;
        const wp::int32 var_288 = 2;
        wp::int32 var_289;
        bool var_290;
        bool var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::int32 var_298;
        wp::int32* var_299;
        const wp::int32 var_300 = 0;
        bool var_301;
        wp::int32 var_302;
        const wp::int32 var_303 = 1;
        wp::int32 var_304;
        const wp::int32 var_305 = 1;
        const wp::int32 var_306 = -1;
        const wp::int32 var_307 = 2;
        wp::range_t var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        wp::int32 var_312;
        const wp::int32 var_313 = 0;
        bool var_314;
        const wp::int32 var_315 = 0;
        bool var_316;
        const wp::int32 var_317 = 0;
        bool var_318;
        bool var_319;
        wp::int32 var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        bool var_323;
        bool var_324;
        const wp::int32 var_325 = 2;
        wp::int32 var_326;
        bool var_327;
        bool var_328;
        wp::int32 var_329;
        wp::int32 var_330;
        wp::int32 var_331;
        wp::int32 var_332;
        wp::int32 var_333;
        wp::int32 var_334;
        wp::int32 var_335;
        wp::int32* var_336;
        const wp::int32 var_337 = 0;
        bool var_338;
        wp::int32 var_339;
        const wp::int32 var_340 = 1;
        wp::int32 var_341;
        const wp::int32 var_342 = 0;
        bool var_343;
        const wp::int32 var_344 = 1;
        //---------
        // forward
        // def k_expand_leaf_flags(                                                               <L 324>
        // tid = wp.tid()                                                                         <L 330>
        var_0 = builtin_tid1d();
        // nx = leaves_dim[0]                                                                     <L 331>
        var_2 = wp::extract(var_leaves_dim, var_1);
        // ny = leaves_dim[1]                                                                     <L 332>
        var_4 = wp::extract(var_leaves_dim, var_3);
        // x = tid % nx                                                                           <L 333>
        var_5 = wp::mod(var_0, var_2);
        // y = (tid // nx) % ny                                                                   <L 334>
        var_6 = wp::floordiv(var_0, var_2);
        var_7 = wp::mod(var_6, var_4);
        // z = tid // (nx * ny)                                                                   <L 335>
        var_8 = wp::mul(var_2, var_4);
        var_9 = wp::floordiv(var_0, var_8);
        // flagged = wp.int32(0)                                                                  <L 336>
        var_11 = wp::int32(var_10);
        // for dz in range(-1, 2):                                                                <L 337>
        // for dy in range(-1, 2):                                                                <L 338>
        // for dx in range(-1, 2):                                                                <L 339>
        var_16 = wp::range(var_14, var_15);
        start_for_0:;
            if (iter_cmp(var_16) == 0) goto end_for_0;
            var_17 = wp::iter_next(var_16);
            // xx = x + dx                                                                        <L 340>
            var_18 = wp::add(var_5, var_17);
            // yy = y + dy                                                                        <L 341>
            var_19 = wp::add(var_7, var_13);
            // zz = z + dz                                                                        <L 342>
            var_20 = wp::add(var_9, var_12);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_22 = (var_18 < var_21);
            var_24 = (var_19 < var_23);
            var_26 = (var_20 < var_25);
            var_27 = var_22 || var_24 || var_26;
            if (var_27) {
                // continue                                                                       <L 344>
                goto start_for_0;
            }
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_28 = (var_18 >= var_2);
            var_29 = (var_19 >= var_4);
            var_31 = wp::extract(var_leaves_dim, var_30);
            var_32 = (var_20 >= var_31);
            var_33 = var_28 || var_29 || var_32;
            if (var_33) {
                // continue                                                                       <L 346>
                goto start_for_0;
            }
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_34 = wp::mul(var_4, var_20);
            var_35 = wp::add(var_19, var_34);
            var_36 = wp::mul(var_2, var_35);
            var_37 = wp::add(var_18, var_36);
            var_38 = wp::address(var_leaf_flags_in, var_37);
            var_41 = wp::load(var_38);
            var_40 = (var_41 != var_39);
            if (var_40) {
                // flagged = 1                                                                    <L 348>
            }
            var_43 = wp::where(var_40, var_42, var_11);
            wp::assign(var_11, var_43);
            goto start_for_0;
        end_for_0:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_47 = wp::range(var_45, var_46);
        start_for_2:;
            if (iter_cmp(var_47) == 0) goto end_for_2;
            var_48 = wp::iter_next(var_47);
            // xx = x + dx                                                                        <L 340>
            var_49 = wp::add(var_5, var_48);
            // yy = y + dy                                                                        <L 341>
            var_50 = wp::add(var_7, var_44);
            // zz = z + dz                                                                        <L 342>
            var_51 = wp::add(var_9, var_12);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_53 = (var_49 < var_52);
            var_55 = (var_50 < var_54);
            var_57 = (var_51 < var_56);
            var_58 = var_53 || var_55 || var_57;
            if (var_58) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_49);
                wp::assign(var_19, var_50);
                wp::assign(var_20, var_51);
                goto start_for_2;
            }
            var_59 = wp::where(var_58, var_18, var_49);
            var_60 = wp::where(var_58, var_19, var_50);
            var_61 = wp::where(var_58, var_20, var_51);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_62 = (var_59 >= var_2);
            var_63 = (var_60 >= var_4);
            var_65 = wp::extract(var_leaves_dim, var_64);
            var_66 = (var_61 >= var_65);
            var_67 = var_62 || var_63 || var_66;
            if (var_67) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_59);
                wp::assign(var_19, var_60);
                wp::assign(var_20, var_61);
                goto start_for_2;
            }
            var_68 = wp::where(var_67, var_18, var_59);
            var_69 = wp::where(var_67, var_19, var_60);
            var_70 = wp::where(var_67, var_20, var_61);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_71 = wp::mul(var_4, var_70);
            var_72 = wp::add(var_69, var_71);
            var_73 = wp::mul(var_2, var_72);
            var_74 = wp::add(var_68, var_73);
            var_75 = wp::address(var_leaf_flags_in, var_74);
            var_78 = wp::load(var_75);
            var_77 = (var_78 != var_76);
            if (var_77) {
                // flagged = 1                                                                    <L 348>
            }
            var_80 = wp::where(var_77, var_79, var_11);
            wp::assign(var_11, var_80);
            wp::assign(var_18, var_68);
            wp::assign(var_19, var_69);
            wp::assign(var_20, var_70);
            goto start_for_2;
        end_for_2:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_84 = wp::range(var_82, var_83);
        start_for_4:;
            if (iter_cmp(var_84) == 0) goto end_for_4;
            var_85 = wp::iter_next(var_84);
            // xx = x + dx                                                                        <L 340>
            var_86 = wp::add(var_5, var_85);
            // yy = y + dy                                                                        <L 341>
            var_87 = wp::add(var_7, var_81);
            // zz = z + dz                                                                        <L 342>
            var_88 = wp::add(var_9, var_12);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_90 = (var_86 < var_89);
            var_92 = (var_87 < var_91);
            var_94 = (var_88 < var_93);
            var_95 = var_90 || var_92 || var_94;
            if (var_95) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_86);
                wp::assign(var_19, var_87);
                wp::assign(var_20, var_88);
                goto start_for_4;
            }
            var_96 = wp::where(var_95, var_18, var_86);
            var_97 = wp::where(var_95, var_19, var_87);
            var_98 = wp::where(var_95, var_20, var_88);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_99 = (var_96 >= var_2);
            var_100 = (var_97 >= var_4);
            var_102 = wp::extract(var_leaves_dim, var_101);
            var_103 = (var_98 >= var_102);
            var_104 = var_99 || var_100 || var_103;
            if (var_104) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_96);
                wp::assign(var_19, var_97);
                wp::assign(var_20, var_98);
                goto start_for_4;
            }
            var_105 = wp::where(var_104, var_18, var_96);
            var_106 = wp::where(var_104, var_19, var_97);
            var_107 = wp::where(var_104, var_20, var_98);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_108 = wp::mul(var_4, var_107);
            var_109 = wp::add(var_106, var_108);
            var_110 = wp::mul(var_2, var_109);
            var_111 = wp::add(var_105, var_110);
            var_112 = wp::address(var_leaf_flags_in, var_111);
            var_115 = wp::load(var_112);
            var_114 = (var_115 != var_113);
            if (var_114) {
                // flagged = 1                                                                    <L 348>
            }
            var_117 = wp::where(var_114, var_116, var_11);
            wp::assign(var_11, var_117);
            wp::assign(var_18, var_105);
            wp::assign(var_19, var_106);
            wp::assign(var_20, var_107);
            goto start_for_4;
        end_for_4:;
        // for dy in range(-1, 2):                                                                <L 338>
        // for dx in range(-1, 2):                                                                <L 339>
        var_122 = wp::range(var_120, var_121);
        start_for_6:;
            if (iter_cmp(var_122) == 0) goto end_for_6;
            var_123 = wp::iter_next(var_122);
            // xx = x + dx                                                                        <L 340>
            var_124 = wp::add(var_5, var_123);
            // yy = y + dy                                                                        <L 341>
            var_125 = wp::add(var_7, var_119);
            // zz = z + dz                                                                        <L 342>
            var_126 = wp::add(var_9, var_118);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_128 = (var_124 < var_127);
            var_130 = (var_125 < var_129);
            var_132 = (var_126 < var_131);
            var_133 = var_128 || var_130 || var_132;
            if (var_133) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_124);
                wp::assign(var_19, var_125);
                wp::assign(var_20, var_126);
                goto start_for_6;
            }
            var_134 = wp::where(var_133, var_18, var_124);
            var_135 = wp::where(var_133, var_19, var_125);
            var_136 = wp::where(var_133, var_20, var_126);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_137 = (var_134 >= var_2);
            var_138 = (var_135 >= var_4);
            var_140 = wp::extract(var_leaves_dim, var_139);
            var_141 = (var_136 >= var_140);
            var_142 = var_137 || var_138 || var_141;
            if (var_142) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_134);
                wp::assign(var_19, var_135);
                wp::assign(var_20, var_136);
                goto start_for_6;
            }
            var_143 = wp::where(var_142, var_18, var_134);
            var_144 = wp::where(var_142, var_19, var_135);
            var_145 = wp::where(var_142, var_20, var_136);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_146 = wp::mul(var_4, var_145);
            var_147 = wp::add(var_144, var_146);
            var_148 = wp::mul(var_2, var_147);
            var_149 = wp::add(var_143, var_148);
            var_150 = wp::address(var_leaf_flags_in, var_149);
            var_153 = wp::load(var_150);
            var_152 = (var_153 != var_151);
            if (var_152) {
                // flagged = 1                                                                    <L 348>
            }
            var_155 = wp::where(var_152, var_154, var_11);
            wp::assign(var_11, var_155);
            wp::assign(var_18, var_143);
            wp::assign(var_19, var_144);
            wp::assign(var_20, var_145);
            goto start_for_6;
        end_for_6:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_159 = wp::range(var_157, var_158);
        start_for_8:;
            if (iter_cmp(var_159) == 0) goto end_for_8;
            var_160 = wp::iter_next(var_159);
            // xx = x + dx                                                                        <L 340>
            var_161 = wp::add(var_5, var_160);
            // yy = y + dy                                                                        <L 341>
            var_162 = wp::add(var_7, var_156);
            // zz = z + dz                                                                        <L 342>
            var_163 = wp::add(var_9, var_118);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_165 = (var_161 < var_164);
            var_167 = (var_162 < var_166);
            var_169 = (var_163 < var_168);
            var_170 = var_165 || var_167 || var_169;
            if (var_170) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_161);
                wp::assign(var_19, var_162);
                wp::assign(var_20, var_163);
                goto start_for_8;
            }
            var_171 = wp::where(var_170, var_18, var_161);
            var_172 = wp::where(var_170, var_19, var_162);
            var_173 = wp::where(var_170, var_20, var_163);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_174 = (var_171 >= var_2);
            var_175 = (var_172 >= var_4);
            var_177 = wp::extract(var_leaves_dim, var_176);
            var_178 = (var_173 >= var_177);
            var_179 = var_174 || var_175 || var_178;
            if (var_179) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_171);
                wp::assign(var_19, var_172);
                wp::assign(var_20, var_173);
                goto start_for_8;
            }
            var_180 = wp::where(var_179, var_18, var_171);
            var_181 = wp::where(var_179, var_19, var_172);
            var_182 = wp::where(var_179, var_20, var_173);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_183 = wp::mul(var_4, var_182);
            var_184 = wp::add(var_181, var_183);
            var_185 = wp::mul(var_2, var_184);
            var_186 = wp::add(var_180, var_185);
            var_187 = wp::address(var_leaf_flags_in, var_186);
            var_190 = wp::load(var_187);
            var_189 = (var_190 != var_188);
            if (var_189) {
                // flagged = 1                                                                    <L 348>
            }
            var_192 = wp::where(var_189, var_191, var_11);
            wp::assign(var_11, var_192);
            wp::assign(var_18, var_180);
            wp::assign(var_19, var_181);
            wp::assign(var_20, var_182);
            goto start_for_8;
        end_for_8:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_196 = wp::range(var_194, var_195);
        start_for_10:;
            if (iter_cmp(var_196) == 0) goto end_for_10;
            var_197 = wp::iter_next(var_196);
            // xx = x + dx                                                                        <L 340>
            var_198 = wp::add(var_5, var_197);
            // yy = y + dy                                                                        <L 341>
            var_199 = wp::add(var_7, var_193);
            // zz = z + dz                                                                        <L 342>
            var_200 = wp::add(var_9, var_118);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_202 = (var_198 < var_201);
            var_204 = (var_199 < var_203);
            var_206 = (var_200 < var_205);
            var_207 = var_202 || var_204 || var_206;
            if (var_207) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_198);
                wp::assign(var_19, var_199);
                wp::assign(var_20, var_200);
                goto start_for_10;
            }
            var_208 = wp::where(var_207, var_18, var_198);
            var_209 = wp::where(var_207, var_19, var_199);
            var_210 = wp::where(var_207, var_20, var_200);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_211 = (var_208 >= var_2);
            var_212 = (var_209 >= var_4);
            var_214 = wp::extract(var_leaves_dim, var_213);
            var_215 = (var_210 >= var_214);
            var_216 = var_211 || var_212 || var_215;
            if (var_216) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_208);
                wp::assign(var_19, var_209);
                wp::assign(var_20, var_210);
                goto start_for_10;
            }
            var_217 = wp::where(var_216, var_18, var_208);
            var_218 = wp::where(var_216, var_19, var_209);
            var_219 = wp::where(var_216, var_20, var_210);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_220 = wp::mul(var_4, var_219);
            var_221 = wp::add(var_218, var_220);
            var_222 = wp::mul(var_2, var_221);
            var_223 = wp::add(var_217, var_222);
            var_224 = wp::address(var_leaf_flags_in, var_223);
            var_227 = wp::load(var_224);
            var_226 = (var_227 != var_225);
            if (var_226) {
                // flagged = 1                                                                    <L 348>
            }
            var_229 = wp::where(var_226, var_228, var_11);
            wp::assign(var_11, var_229);
            wp::assign(var_18, var_217);
            wp::assign(var_19, var_218);
            wp::assign(var_20, var_219);
            goto start_for_10;
        end_for_10:;
        // for dy in range(-1, 2):                                                                <L 338>
        // for dx in range(-1, 2):                                                                <L 339>
        var_234 = wp::range(var_232, var_233);
        start_for_12:;
            if (iter_cmp(var_234) == 0) goto end_for_12;
            var_235 = wp::iter_next(var_234);
            // xx = x + dx                                                                        <L 340>
            var_236 = wp::add(var_5, var_235);
            // yy = y + dy                                                                        <L 341>
            var_237 = wp::add(var_7, var_231);
            // zz = z + dz                                                                        <L 342>
            var_238 = wp::add(var_9, var_230);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_240 = (var_236 < var_239);
            var_242 = (var_237 < var_241);
            var_244 = (var_238 < var_243);
            var_245 = var_240 || var_242 || var_244;
            if (var_245) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_236);
                wp::assign(var_19, var_237);
                wp::assign(var_20, var_238);
                goto start_for_12;
            }
            var_246 = wp::where(var_245, var_18, var_236);
            var_247 = wp::where(var_245, var_19, var_237);
            var_248 = wp::where(var_245, var_20, var_238);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_249 = (var_246 >= var_2);
            var_250 = (var_247 >= var_4);
            var_252 = wp::extract(var_leaves_dim, var_251);
            var_253 = (var_248 >= var_252);
            var_254 = var_249 || var_250 || var_253;
            if (var_254) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_246);
                wp::assign(var_19, var_247);
                wp::assign(var_20, var_248);
                goto start_for_12;
            }
            var_255 = wp::where(var_254, var_18, var_246);
            var_256 = wp::where(var_254, var_19, var_247);
            var_257 = wp::where(var_254, var_20, var_248);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_258 = wp::mul(var_4, var_257);
            var_259 = wp::add(var_256, var_258);
            var_260 = wp::mul(var_2, var_259);
            var_261 = wp::add(var_255, var_260);
            var_262 = wp::address(var_leaf_flags_in, var_261);
            var_265 = wp::load(var_262);
            var_264 = (var_265 != var_263);
            if (var_264) {
                // flagged = 1                                                                    <L 348>
            }
            var_267 = wp::where(var_264, var_266, var_11);
            wp::assign(var_11, var_267);
            wp::assign(var_18, var_255);
            wp::assign(var_19, var_256);
            wp::assign(var_20, var_257);
            goto start_for_12;
        end_for_12:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_271 = wp::range(var_269, var_270);
        start_for_14:;
            if (iter_cmp(var_271) == 0) goto end_for_14;
            var_272 = wp::iter_next(var_271);
            // xx = x + dx                                                                        <L 340>
            var_273 = wp::add(var_5, var_272);
            // yy = y + dy                                                                        <L 341>
            var_274 = wp::add(var_7, var_268);
            // zz = z + dz                                                                        <L 342>
            var_275 = wp::add(var_9, var_230);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_277 = (var_273 < var_276);
            var_279 = (var_274 < var_278);
            var_281 = (var_275 < var_280);
            var_282 = var_277 || var_279 || var_281;
            if (var_282) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_273);
                wp::assign(var_19, var_274);
                wp::assign(var_20, var_275);
                goto start_for_14;
            }
            var_283 = wp::where(var_282, var_18, var_273);
            var_284 = wp::where(var_282, var_19, var_274);
            var_285 = wp::where(var_282, var_20, var_275);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_286 = (var_283 >= var_2);
            var_287 = (var_284 >= var_4);
            var_289 = wp::extract(var_leaves_dim, var_288);
            var_290 = (var_285 >= var_289);
            var_291 = var_286 || var_287 || var_290;
            if (var_291) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_283);
                wp::assign(var_19, var_284);
                wp::assign(var_20, var_285);
                goto start_for_14;
            }
            var_292 = wp::where(var_291, var_18, var_283);
            var_293 = wp::where(var_291, var_19, var_284);
            var_294 = wp::where(var_291, var_20, var_285);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_295 = wp::mul(var_4, var_294);
            var_296 = wp::add(var_293, var_295);
            var_297 = wp::mul(var_2, var_296);
            var_298 = wp::add(var_292, var_297);
            var_299 = wp::address(var_leaf_flags_in, var_298);
            var_302 = wp::load(var_299);
            var_301 = (var_302 != var_300);
            if (var_301) {
                // flagged = 1                                                                    <L 348>
            }
            var_304 = wp::where(var_301, var_303, var_11);
            wp::assign(var_11, var_304);
            wp::assign(var_18, var_292);
            wp::assign(var_19, var_293);
            wp::assign(var_20, var_294);
            goto start_for_14;
        end_for_14:;
        // for dx in range(-1, 2):                                                                <L 339>
        var_308 = wp::range(var_306, var_307);
        start_for_16:;
            if (iter_cmp(var_308) == 0) goto end_for_16;
            var_309 = wp::iter_next(var_308);
            // xx = x + dx                                                                        <L 340>
            var_310 = wp::add(var_5, var_309);
            // yy = y + dy                                                                        <L 341>
            var_311 = wp::add(var_7, var_305);
            // zz = z + dz                                                                        <L 342>
            var_312 = wp::add(var_9, var_230);
            // if xx < 0 or yy < 0 or zz < 0:                                                     <L 343>
            var_314 = (var_310 < var_313);
            var_316 = (var_311 < var_315);
            var_318 = (var_312 < var_317);
            var_319 = var_314 || var_316 || var_318;
            if (var_319) {
                // continue                                                                       <L 344>
                wp::assign(var_18, var_310);
                wp::assign(var_19, var_311);
                wp::assign(var_20, var_312);
                goto start_for_16;
            }
            var_320 = wp::where(var_319, var_18, var_310);
            var_321 = wp::where(var_319, var_19, var_311);
            var_322 = wp::where(var_319, var_20, var_312);
            // if xx >= nx or yy >= ny or zz >= leaves_dim[2]:                                    <L 345>
            var_323 = (var_320 >= var_2);
            var_324 = (var_321 >= var_4);
            var_326 = wp::extract(var_leaves_dim, var_325);
            var_327 = (var_322 >= var_326);
            var_328 = var_323 || var_324 || var_327;
            if (var_328) {
                // continue                                                                       <L 346>
                wp::assign(var_18, var_320);
                wp::assign(var_19, var_321);
                wp::assign(var_20, var_322);
                goto start_for_16;
            }
            var_329 = wp::where(var_328, var_18, var_320);
            var_330 = wp::where(var_328, var_19, var_321);
            var_331 = wp::where(var_328, var_20, var_322);
            // if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:                                   <L 347>
            var_332 = wp::mul(var_4, var_331);
            var_333 = wp::add(var_330, var_332);
            var_334 = wp::mul(var_2, var_333);
            var_335 = wp::add(var_329, var_334);
            var_336 = wp::address(var_leaf_flags_in, var_335);
            var_339 = wp::load(var_336);
            var_338 = (var_339 != var_337);
            if (var_338) {
                // flagged = 1                                                                    <L 348>
            }
            var_341 = wp::where(var_338, var_340, var_11);
            wp::assign(var_11, var_341);
            wp::assign(var_18, var_329);
            wp::assign(var_19, var_330);
            wp::assign(var_20, var_331);
            goto start_for_16;
        end_for_16:;
        // if flagged != 0:                                                                       <L 349>
        var_343 = (var_11 != var_342);
        if (var_343) {
            // leaf_flags_out[tid] = 1                                                            <L 350>
            wp::array_store(var_leaf_flags_out, var_0, var_344);
        }
    }
}



extern "C" __global__ void k_allocate_leaves_e70872b5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_leaf_flags,
    wp::vec_t<3, wp::int32> var_leaves_dim,
    wp::int32 var_capacity,
    wp::array_t<wp::int32> var_leaf_table,
    wp::array_t<wp::vec_t<3, wp::int32>> var_leaf_coords,
    wp::array_t<wp::int32> var_num_leaves,
    wp::array_t<wp::int32> var_overflow)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 0;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        bool var_12;
        const wp::int32 var_13 = 1;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::int32> var_24;
        //---------
        // forward
        // def k_allocate_leaves(                                                                 <L 354>
        // tid = wp.tid()                                                                         <L 363>
        var_0 = builtin_tid1d();
        // if leaf_flags[tid] == 0:                                                               <L 364>
        var_1 = wp::address(var_leaf_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 365>
            continue;
        }
        // if leaf_table[tid] >= 0:                                                               <L 366>
        var_5 = wp::address(var_leaf_table, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 >= var_6);
        if (var_7) {
            // return                                                                             <L 367>
            continue;
        }
        // slot = wp.atomic_add(num_leaves, 0, 1)                                                 <L 368>
        var_11 = wp::atomic_add(var_num_leaves, var_9, var_10);
        // if slot >= capacity:                                                                   <L 369>
        var_12 = (var_11 >= var_capacity);
        if (var_12) {
            // overflow[0] = 1                                                                    <L 370>
            wp::array_store(var_overflow, var_14, var_13);
            // return                                                                             <L 371>
            continue;
        }
        // leaf_table[tid] = slot                                                                 <L 372>
        wp::array_store(var_leaf_table, var_0, var_11);
        // nx = leaves_dim[0]                                                                     <L 373>
        var_16 = wp::extract(var_leaves_dim, var_15);
        // ny = leaves_dim[1]                                                                     <L 374>
        var_18 = wp::extract(var_leaves_dim, var_17);
        // leaf_coords[slot] = wp.vec3i(tid % nx, (tid // nx) % ny, tid // (nx * ny))             <L 375>
        var_19 = wp::mod(var_0, var_16);
        var_20 = wp::floordiv(var_0, var_16);
        var_21 = wp::mod(var_20, var_18);
        var_22 = wp::mul(var_16, var_18);
        var_23 = wp::floordiv(var_0, var_22);
        var_24 = wp::vec_t<3, wp::int32>(var_19, var_21, var_23);
        wp::array_store(var_leaf_coords, var_11, var_24);
    }
}



extern "C" __global__ void k_clear_leaf_topology_ff10827b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_leaf_table,
    wp::array_t<wp::int32> var_leaf_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = -1;
        const wp::int32 var_2 = 0;
        //---------
        // forward
        // def k_clear_leaf_topology(                                                             <L 266>
        // tid = wp.tid()                                                                         <L 270>
        var_0 = builtin_tid1d();
        // leaf_table[tid] = -1                                                                   <L 271>
        wp::array_store(var_leaf_table, var_0, var_1);
        // leaf_flags[tid] = 0                                                                    <L 272>
        wp::array_store(var_leaf_flags, var_0, var_2);
    }
}



extern "C" __global__ void k_mark_leaves_from_points_c1d83ceb_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_points,
    wp::array_t<wp::int32> var_count,
    wp::vec_t<3, wp::float32> var_origin,
    wp::float32 var_voxel_size,
    wp::vec_t<3, wp::int32> var_voxels_dim,
    wp::vec_t<3, wp::int32> var_leaves_dim,
    wp::array_t<wp::int32> var_leaf_flags)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::int32> var_8;
        bool var_9;
        bool var_10;
        wp::vec_t<3, wp::int32> var_11;
        const wp::int32 var_12 = 1;
        wp::int32 var_13;
        //---------
        // forward
        // def k_mark_leaves_from_points(                                                         <L 276>
        // tid = wp.tid()                                                                         <L 285>
        var_0 = builtin_tid1d();
        // if tid >= count[0]:                                                                    <L 286>
        var_2 = wp::address(var_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 287>
            continue;
        }
        // vp = world_to_voxel(points[tid], origin, voxel_size)                                   <L 288>
        var_5 = wp::address(var_points, var_0);
        var_7 = wp::load(var_5);
        var_6 = world_to_voxel_0(var_7, var_origin, var_voxel_size);
        // ijk = voxel_floor(vp)                                                                  <L 289>
        var_8 = voxel_floor_0(var_6);
        // if not voxel_in_domain(ijk, voxels_dim):                                               <L 290>
        var_9 = voxel_in_domain_0(var_8, var_voxels_dim);
        var_10 = wp::unot(var_9);
        if (var_10) {
            // return                                                                             <L 291>
            continue;
        }
        // lc = leaf_coord_of_voxel(ijk)                                                          <L 292>
        var_11 = leaf_coord_of_voxel_0(var_8);
        // leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1                                      <L 293>
        var_13 = leaf_linear_index_0(var_11, var_leaves_dim);
        wp::array_store(var_leaf_flags, var_13, var_12);
    }
}

