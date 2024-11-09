#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cmath>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define NUM_WARPS (BLOCK_SIZE/32)

#define CHECK_VALID_NUMBER(A) assert(!std::isnan(A) && !std::isinf(A))

__forceinline__ __device__ void getRect(const float3 p, float max_radius, uint3& rect_min, uint3& rect_max, dim3 grid, const float3 voxel_physical, const float3 offset)
{
    float BX, BY, BZ;
    BX = BLOCK_X * voxel_physical.x;
    BY = BLOCK_Y * voxel_physical.y;
    BZ = BLOCK_Z * voxel_physical.z;
    
	rect_min = {
		min(grid.x, max((int)0, (int)floor((p.x + offset.x - max_radius) / BX))),
		min(grid.y, max((int)0, (int)floor((p.y + offset.y - max_radius) / BY))),
		min(grid.z, max((int)0, (int)floor((p.z + offset.z - max_radius) / BZ)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)ceil((p.x + offset.x + max_radius) / BX))),
		min(grid.y, max((int)0, (int)ceil((p.y + offset.y + max_radius) / BY))),
		min(grid.z, max((int)0, (int)ceil((p.z + offset.z + max_radius) / BZ))),
	};
}



#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
