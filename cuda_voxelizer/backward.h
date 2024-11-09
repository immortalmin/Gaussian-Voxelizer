#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace BACKWARD
{
	void voxelize(
        const dim3 grid, 
        const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        const float3 voxel_physical,
        const float3 volume_pixel,
        const float3 volume_physical,
        const float* means3D,
        const float* densities,
        const float* cov3Ds_inv,
        const float* dL_daccum,
        float3* dL_dmeans3D,
        float* dL_ddensities,
        float* dL_dcov3Ds_inv 
    );

	void preprocess(
        int P,
        const float* radii,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float* cov3Ds,
        const float* dL_dcov3Ds_inv,
        float* dL_dcov3Ds,
        glm::vec3* dL_dscales,
        glm::vec4* dL_drotations
    );
}

#endif
