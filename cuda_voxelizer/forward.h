#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD{
    void preprocess(
        int P,
	    const float* means3D,
        const float* densities,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        float* cov3Ds,
        float* cov3Ds_inv,
        float* radii,
        const dim3 grid,
        const float3 voxel_physical,
        const float3 volume_offset,
        const float3 volume_center,
        uint32_t* blocks_touched
    );

    void voxelize(
        const dim3 grid,
        const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        const float3 voxel_physical,
        const float3 volume_pixel,
        const float3 volume_physical,
        const float3 volume_center,
	    const float* means3D,
        const float* densities,
        const float* cov3Ds_inv,
    	float* volume
    );

}

#endif
