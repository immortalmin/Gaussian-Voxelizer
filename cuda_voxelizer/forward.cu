#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "helper_math.h"
#include <cassert>
#include <cmath>
#include <complex>
namespace cg = cooperative_groups;

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ void computeCov3DInv(const float* cov3D, float* cov3D_inv) {
    float det = cov3D[0] * (cov3D[3] * cov3D[5] - cov3D[4] * cov3D[4]) - cov3D[1] * (cov3D[1] * cov3D[5] - cov3D[2] * cov3D[4]) + cov3D[2] * (cov3D[1] * cov3D[4] - cov3D[2] * cov3D[3]);
    //FIXME:要确保行列式不为0
    assert(det != 0.0);
    float det_inv = 1.0 / det;
    cov3D_inv[0] = det_inv * (cov3D[3] * cov3D[5] - cov3D[4] * cov3D[4]);
    cov3D_inv[1] = det_inv * (cov3D[2] * cov3D[4] - cov3D[1] * cov3D[5]);
    cov3D_inv[2] = det_inv * (cov3D[1] * cov3D[4] - cov3D[2] * cov3D[3]);
    cov3D_inv[3] = det_inv * (cov3D[0] * cov3D[5] - cov3D[2] * cov3D[2]);
    cov3D_inv[4] = det_inv * (cov3D[1] * cov3D[2] - cov3D[0] * cov3D[4]);
    cov3D_inv[5] = det_inv * (cov3D[0] * cov3D[3] - cov3D[1] * cov3D[1]);
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
voxelizeCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float3 __restrict__ voxel_physical,
    const float3 __restrict__ volume_pixel,
    const float3 __restrict__ volume_physical,
    const float3 __restrict__ volume_center,
	const float* __restrict__ means3D,
	const float* __restrict__ densities,
	const float* __restrict__ cov3Ds_inv,
	float* __restrict__ volume
){
    auto block = cg::this_thread_block();
    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    float3 idx_f;
    idx_f.x = static_cast<float>(idx_x);
    idx_f.y = static_cast<float>(idx_y);
    idx_f.z = static_cast<float>(idx_z);

    bool inside = idx_x < volume_pixel.x && idx_y < volume_pixel.y && idx_z < volume_pixel.z;
    // the index of the voxel in the volume.
    int globalIdx = idx_x * volume_pixel.y * volume_pixel.z + idx_y * volume_pixel.z + idx_z;

    float3 start = (voxel_physical - volume_physical) / 2.0 + volume_center;
    // x is the physical position of the center of the voxel.
    float3 x = start + idx_f * voxel_physical;

    uint2 range = ranges[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;
    
    __shared__ float3 collected_xyz[BLOCK_SIZE];
    __shared__ float collected_cov3Ds_inv[BLOCK_SIZE][6];
    __shared__ float collected_density[BLOCK_SIZE];

    float accum = 0.0f;

    for(int i=0;i<rounds;++i, toDo-=BLOCK_SIZE) {
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if(range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_xyz[block.thread_rank()] = {means3D[3 * coll_id], means3D[3 * coll_id + 1], means3D[3 * coll_id + 2]};
            for(int j=0;j<6;++j)
                collected_cov3Ds_inv[block.thread_rank()][j] = cov3Ds_inv[coll_id * 6 + j];
            collected_density[block.thread_rank()] = densities[coll_id];
        }
        block.sync();
        for(int j=0;inside && j<min(BLOCK_SIZE, toDo); ++j) {
            float3 p = collected_xyz[j];
            float3 d = p - x;
            // FIXME: 可以优化一下
            float power = -0.5 * (collected_cov3Ds_inv[j][0] * d.x * d.x + 2.0 * collected_cov3Ds_inv[j][1] * d.x * d.y + 2.0 * collected_cov3Ds_inv[j][2] * d.x * d.z + collected_cov3Ds_inv[j][3] * d.y * d.y + 2.0 * collected_cov3Ds_inv[j][4] * d.y * d.z + collected_cov3Ds_inv[j][5] * d.z * d.z);
            if (power > 0.0f) continue;
            accum += collected_density[j] * exp(power);
        }
    }
    if(inside) volume[globalIdx] = accum;
}

__global__ void preprocessCUDA(
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
){
    auto idx = cg::this_grid().thread_rank();
    if(idx >= P) return;

    radii[idx] = 0.0f;
    blocks_touched[idx] = 0;

    computeCov3D(scales[idx], rotations[idx], cov3Ds + idx * 6);
    computeCov3DInv(cov3Ds + idx * 6, cov3Ds_inv + idx * 6);

    float max_scale = max(scales[idx].x, max(scales[idx].y, scales[idx].z));
    float radius = 3.0 * max_scale;
    if(radius == 0.0) return;

    const float3 point = {means3D[idx*3], means3D[idx*3+1], means3D[idx*3+2]};
    uint3 rect_min, rect_max;
    getRect(point, radius, rect_min, rect_max, grid, voxel_physical, volume_offset, volume_center);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0) return;

    radii[idx] = radius;
    blocks_touched[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z);
}

void FORWARD::voxelize(
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
) {
	voxelizeCUDA<<<grid, block>>>(
        ranges,
        point_list,
        voxel_physical,
        volume_pixel,
        volume_physical,
        volume_center,
		means3D,
		densities,
        cov3Ds_inv,
		volume
	);
}

void FORWARD::preprocess(
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
) {
    preprocessCUDA<<<(P + 255) / 256, 256>>>(
        P,
        means3D,
        densities,
        scales,
        rotations,
        cov3Ds,
        cov3Ds_inv,
        radii,
        grid,
        voxel_physical,
        volume_offset,
        volume_center,
        blocks_touched
    );
}
