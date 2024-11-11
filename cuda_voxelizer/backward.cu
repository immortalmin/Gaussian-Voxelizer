#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "helper_math.h"
namespace cg = cooperative_groups;

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale.x;
	dL_dMt[1] *= scale.y;
	dL_dMt[2] *= scale.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// 原版代码，甚至有错误，但是有时指标反而更高，所以先保留着
__device__ void _computeCov3DInv(const float* cov3D, const float* dL_dcov3D_inv, float* dL_dcov3D) {
    glm::mat3 _dL_dcov3D_inv_matrix(dL_dcov3D_inv[0], dL_dcov3D_inv[1], dL_dcov3D_inv[2], 
                                   dL_dcov3D_inv[1], dL_dcov3D_inv[3], dL_dcov3D_inv[4],
                                   dL_dcov3D_inv[2], dL_dcov3D_inv[4], dL_dcov3D_inv[5]);
    float* dL_dcov3D_inv_matrix = glm::value_ptr(_dL_dcov3D_inv_matrix);
    glm::mat3 _dL_dcov3D_matrix(0.0f);
    float* dL_dcov3D_matrix = glm::value_ptr(_dL_dcov3D_matrix);
    glm::mat3 dcov3Dstar_dcov3D[9];
    glm::mat3 _cov3D_matrix(cov3D[0], cov3D[1], cov3D[2], 
                           cov3D[1], cov3D[3], cov3D[4],
                           cov3D[2], cov3D[4], cov3D[5]);
    float* cov3D_matrix = glm::value_ptr(_cov3D_matrix);

    //FIXME: 可以在前向传播中保存det，避免重复计算
    float det = cov3D_matrix[0] * (cov3D_matrix[4] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[7]) - cov3D_matrix[1] * (cov3D_matrix[3] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[6]) + cov3D_matrix[2] * (cov3D_matrix[3] * cov3D_matrix[7] - cov3D_matrix[4] * cov3D_matrix[6]);
    
    dcov3Dstar_dcov3D[0] = glm::mat3(0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[8], -cov3D_matrix[7], 0.0f, -cov3D_matrix[5], cov3D_matrix[4]);
    dcov3Dstar_dcov3D[1] = glm::mat3(0.0f, 0.0f, 0.0f, -cov3D_matrix[8], 0.0f, cov3D_matrix[6], cov3D_matrix[5], 0.0f, -cov3D_matrix[3]);
    dcov3Dstar_dcov3D[2] = glm::mat3(0.0f, 0.0f, 0.0f, cov3D_matrix[7], -cov3D_matrix[6], 0.0f, -cov3D_matrix[4], cov3D_matrix[3], 0.0f);
    dcov3Dstar_dcov3D[3] = glm::mat3(0.0f, -cov3D_matrix[8], cov3D_matrix[7], 0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[2], -cov3D_matrix[1]);
    dcov3Dstar_dcov3D[4] = glm::mat3(cov3D_matrix[8], 0.0f, -cov3D_matrix[6], 0.0f, 0.0f, 0.0f, -cov3D_matrix[2], 0.0f, cov3D_matrix[0]);
    dcov3Dstar_dcov3D[5] = glm::mat3(-cov3D_matrix[7], cov3D_matrix[6], 0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[1], -cov3D_matrix[0], 0.0f);
    dcov3Dstar_dcov3D[6] = glm::mat3(0.0f, cov3D_matrix[5], -cov3D_matrix[4], 0.0f, -cov3D_matrix[2], cov3D_matrix[1], 0.0f, 0.0f, 0.0f);
    dcov3Dstar_dcov3D[7] = glm::mat3(-cov3D_matrix[5], 0.0f, cov3D_matrix[3], cov3D_matrix[2], 0.0f, -cov3D_matrix[0], 0.0f, 0.0f, 0.0f);
    dcov3Dstar_dcov3D[8] = glm::mat3(cov3D_matrix[4], -cov3D_matrix[3], 0.0f, -cov3D_matrix[1], cov3D_matrix[0], 0.0f, 0.0f, 0.0f, 0.0f);
    
    glm::mat3 _cov3Dstar(cov3D_matrix[4] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[7], cov3D_matrix[5] * cov3D_matrix[6] - cov3D_matrix[3] * cov3D_matrix[8], cov3D_matrix[3] * cov3D_matrix[7] - cov3D_matrix[4] * cov3D_matrix[6], 
                         cov3D_matrix[2] * cov3D_matrix[7] - cov3D_matrix[1] * cov3D_matrix[8], cov3D_matrix[0] * cov3D_matrix[8] - cov3D_matrix[2] * cov3D_matrix[6], cov3D_matrix[1] * cov3D_matrix[6] - cov3D_matrix[0] * cov3D_matrix[7],
                         cov3D_matrix[1] * cov3D_matrix[5] - cov3D_matrix[2] * cov3D_matrix[4], cov3D_matrix[2] * cov3D_matrix[3] - cov3D_matrix[0] * cov3D_matrix[5], cov3D_matrix[0] * cov3D_matrix[4] - cov3D_matrix[1] * cov3D_matrix[3]);
    float* cov3Dstar = glm::value_ptr(_cov3Dstar);
    
    float det_inv = 1.0f / det;
    float det_square_inv = det_inv * det_inv;
    if(isnan(det_inv) || isinf(det_inv) || isnan(det_square_inv) || isinf(det_square_inv)) return;
    float tmp = 0.0f;
    for(int j=0;j<9;++j) {
        //不需要算9个，只需要算其中6个
        //if(j == 3 || j == 6 || j == 7) continue;
        // 有错误的地方，这里本该重置tmp为0的。但是不重置有时反而指标更高
        // tmp = 0.0f;
        float* dcov3Dstar_dcov3D_j = glm::value_ptr(dcov3Dstar_dcov3D[j]);
        for(int i=0;i<9;++i) {
            tmp += dL_dcov3D_inv_matrix[i] * dcov3Dstar_dcov3D_j[i];
        }
        dL_dcov3D_matrix[j] = det_inv * tmp;
        tmp = 0.0;
        for(int i=0;i<9;++i) {
            tmp += dL_dcov3D_inv_matrix[i] * cov3Dstar[i];
        }
        dL_dcov3D_matrix[j] -= det_square_inv * cov3Dstar[j] * tmp;
    }

    // 将3乘3矩阵的upper right保存到dL_dcov3D中。
    // 为了和3DGS代码对齐，非对角线元素要乘以2
    dL_dcov3D[0] = dL_dcov3D_matrix[0];
    dL_dcov3D[1] = dL_dcov3D_matrix[1] * 2.0f;
    dL_dcov3D[2] = dL_dcov3D_matrix[2] * 2.0f;
    dL_dcov3D[3] = dL_dcov3D_matrix[4];
    dL_dcov3D[4] = dL_dcov3D_matrix[5] * 2.0f;
    dL_dcov3D[5] = dL_dcov3D_matrix[8];
}

// 代码优化的版本，但是优化后性能并没有什么提升
__device__ void computeCov3DInv(const float* cov3D, const float* dL_dcov3D_inv, float* dL_dcov3D) {
    float dL_dcov3D_inv_matrix[] = {dL_dcov3D_inv[0], dL_dcov3D_inv[1], dL_dcov3D_inv[2], 
                                    dL_dcov3D_inv[1], dL_dcov3D_inv[3], dL_dcov3D_inv[4],
                                    dL_dcov3D_inv[2], dL_dcov3D_inv[4], dL_dcov3D_inv[5]};

    float cov3D_matrix[] = {cov3D[0], cov3D[1], cov3D[2], 
                            cov3D[1], cov3D[3], cov3D[4],
                            cov3D[2], cov3D[4], cov3D[5]};
    float dL_dcov3D_matrix[9] = {0.0};

    //FIXME: 可以在前向传播中保存det，避免重复计算
    float det = cov3D_matrix[0] * (cov3D_matrix[4] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[7]) - cov3D_matrix[1] * (cov3D_matrix[3] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[6]) + cov3D_matrix[2] * (cov3D_matrix[3] * cov3D_matrix[7] - cov3D_matrix[4] * cov3D_matrix[6]);

    float dcov3Dstar_dcov3D[9][9] = {
        {0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[8], -cov3D_matrix[7], 0.0f, -cov3D_matrix[5], cov3D_matrix[4]},
        {0.0f, 0.0f, 0.0f, -cov3D_matrix[8], 0.0f, cov3D_matrix[6], cov3D_matrix[5], 0.0f, -cov3D_matrix[3]},
        {0.0f, 0.0f, 0.0f, cov3D_matrix[7], -cov3D_matrix[6], 0.0f, -cov3D_matrix[4], cov3D_matrix[3], 0.0f},
        {0.0f, -cov3D_matrix[8], cov3D_matrix[7], 0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[2], -cov3D_matrix[1]},
        {cov3D_matrix[8], 0.0f, -cov3D_matrix[6], 0.0f, 0.0f, 0.0f, -cov3D_matrix[2], 0.0f, cov3D_matrix[0]},
        {-cov3D_matrix[7], cov3D_matrix[6], 0.0f, 0.0f, 0.0f, 0.0f, cov3D_matrix[1], -cov3D_matrix[0], 0.0f},
        {0.0f, cov3D_matrix[5], -cov3D_matrix[4], 0.0f, -cov3D_matrix[2], cov3D_matrix[1], 0.0f, 0.0f, 0.0f},
        {-cov3D_matrix[5], 0.0f, cov3D_matrix[3], cov3D_matrix[2], 0.0f, -cov3D_matrix[0], 0.0f, 0.0f, 0.0f},
        {cov3D_matrix[4], -cov3D_matrix[3], 0.0f, -cov3D_matrix[1], cov3D_matrix[0], 0.0f, 0.0f, 0.0f, 0.0f},
    };

    float cov3Dstar[] = {cov3D_matrix[4] * cov3D_matrix[8] - cov3D_matrix[5] * cov3D_matrix[7], cov3D_matrix[5] * cov3D_matrix[6] - cov3D_matrix[3] * cov3D_matrix[8], cov3D_matrix[3] * cov3D_matrix[7] - cov3D_matrix[4] * cov3D_matrix[6], 
                       cov3D_matrix[2] * cov3D_matrix[7] - cov3D_matrix[1] * cov3D_matrix[8], cov3D_matrix[0] * cov3D_matrix[8] - cov3D_matrix[2] * cov3D_matrix[6], cov3D_matrix[1] * cov3D_matrix[6] - cov3D_matrix[0] * cov3D_matrix[7],
                       cov3D_matrix[1] * cov3D_matrix[5] - cov3D_matrix[2] * cov3D_matrix[4], cov3D_matrix[2] * cov3D_matrix[3] - cov3D_matrix[0] * cov3D_matrix[5], cov3D_matrix[0] * cov3D_matrix[4] - cov3D_matrix[1] * cov3D_matrix[3]};
    
    float det_inv = 1.0f / det;
    float det_square_inv = det_inv * det_inv;
    if(isnan(det_inv) || isinf(det_inv) || isnan(det_square_inv) || isinf(det_square_inv)) return;
    // \sigma_{i=0}^8 dL_dcov3D_inv[i] \times cov3Dstar[i]
    float tmp2 = 0.0;
    for(int i=0;i<9;++i) {
        tmp2 += dL_dcov3D_inv_matrix[i] * cov3Dstar[i];
    }
    for(int j=0;j<9;++j) {
        //不需要算9个，只需要算其中6个
        if(j == 3 || j == 6 || j == 7) continue;
        float tmp = 0.0f;
        for(int i=0;i<9;++i) {
            tmp += dL_dcov3D_inv_matrix[i] * dcov3Dstar_dcov3D[j][i];
        }
        dL_dcov3D_matrix[j] = det_inv * tmp - det_square_inv * cov3Dstar[j] * tmp2;
    }

    // 将3乘3矩阵的upper right保存到dL_dcov3D中。
    // 为了和3DGS代码对齐，非对角线元素要乘以2
    dL_dcov3D[0] = dL_dcov3D_matrix[0];
    dL_dcov3D[1] = dL_dcov3D_matrix[1] * 2.0f;
    dL_dcov3D[2] = dL_dcov3D_matrix[2] * 2.0f;
    dL_dcov3D[3] = dL_dcov3D_matrix[4];
    dL_dcov3D[4] = dL_dcov3D_matrix[5] * 2.0f;
    dL_dcov3D[5] = dL_dcov3D_matrix[8];
}

__global__ void preprocessCUDA(
    int P,
    const float* radii,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float* cov3Ds,
    const float* dL_dcov3Ds_inv,
    float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drots
) {
    auto idx = cg::this_grid().thread_rank();
    if(idx >= P || !(radii[idx] > 0)) return;

    computeCov3DInv(cov3Ds + 6 * idx, dL_dcov3Ds_inv + 6 * idx, dL_dcov3Ds + 6 * idx);
    computeCov3D(idx, scales[idx], rotations[idx], dL_dcov3Ds, dL_dscales, dL_drots);
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
    const float* __restrict__ dL_daccums,
    float3* __restrict__ dL_dmeans3D,
    float* __restrict__ dL_ddensities,
    float* __restrict__ dL_dcov3Ds_inv
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
    bool done = !inside;
    // the index of the voxel in the volume.
    int globalIdx = idx_x * volume_pixel.y * volume_pixel.z + idx_y * volume_pixel.z + idx_z;

    float3 start = (voxel_physical - volume_physical) / 2.0 + volume_center;
    // x is the physical position of the center of the voxel.
    float3 x = start + idx_f * voxel_physical;

    uint2 range = ranges[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;
    
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float3 collected_xyz[BLOCK_SIZE];
    __shared__ float collected_cov3Ds_inv[BLOCK_SIZE][6];
    __shared__ float collected_density[BLOCK_SIZE];

    float dL_daccum;
    if(inside) dL_daccum = dL_daccums[globalIdx];

    for(int i=0;i<rounds;++i, toDo-=BLOCK_SIZE) {
        //FIXME: 3DGS在这里也放了一个block.sync()，不知道有何用意
        block.sync();
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if(range.x + progress < range.y) {
            // 3DGS这里是倒序处理，我们不需要。
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xyz[block.thread_rank()] = {means3D[3 * coll_id], means3D[3 * coll_id + 1], means3D[3 * coll_id + 2]};
            for(int j=0;j<6;++j)
                collected_cov3Ds_inv[block.thread_rank()][j] = cov3Ds_inv[coll_id * 6 + j];
            collected_density[block.thread_rank()] = densities[coll_id];
        }
        block.sync();
        for(int j=0;!done && j<min(BLOCK_SIZE, toDo); ++j) {
            // Gaussian的ID
            const int global_id = collected_id[j];
            float3 p = collected_xyz[j];
            float3 d = p - x;
            float power = -0.5 * (collected_cov3Ds_inv[j][0] * d.x * d.x + 2.0 * collected_cov3Ds_inv[j][1] * d.x * d.y + 2.0 * collected_cov3Ds_inv[j][2] * d.x * d.z + collected_cov3Ds_inv[j][3] * d.y * d.y + 2.0 * collected_cov3Ds_inv[j][4] * d.y * d.z + collected_cov3Ds_inv[j][5] * d.z * d.z);
            if (power > 0.0f) continue;
            float exp_power = exp(power);
            float density = collected_density[j];

            float dL_dpower = dL_daccum * density * exp_power;
            float dpower_ddx = -collected_cov3Ds_inv[j][0] * d.x - collected_cov3Ds_inv[j][1] * d.y - collected_cov3Ds_inv[j][2] * d.z;
            float dpower_ddy = -collected_cov3Ds_inv[j][3] * d.y - collected_cov3Ds_inv[j][1] * d.x - collected_cov3Ds_inv[j][4] * d.z;
            float dpower_ddz = -collected_cov3Ds_inv[j][5] * d.z - collected_cov3Ds_inv[j][2] * d.x - collected_cov3Ds_inv[j][4] * d.y;
            atomicAdd(&dL_dmeans3D[global_id].x, dL_dpower * dpower_ddx);
            atomicAdd(&dL_dmeans3D[global_id].y, dL_dpower * dpower_ddy);
            atomicAdd(&dL_dmeans3D[global_id].z, dL_dpower * dpower_ddz);

            atomicAdd(&dL_dcov3Ds_inv[6 * global_id], dL_dpower * (-0.5 * d.x * d.x));
            atomicAdd(&dL_dcov3Ds_inv[6 * global_id + 1], dL_dpower * (-0.5 * d.x * d.y));
            atomicAdd(&dL_dcov3Ds_inv[6 * global_id + 2], dL_dpower * (-0.5 * d.x * d.z));
            atomicAdd(&dL_dcov3Ds_inv[6 * global_id + 3], dL_dpower * (-0.5 * d.y * d.y));
            atomicAdd(&dL_dcov3Ds_inv[6 * global_id + 4], dL_dpower * (-0.5 * d.y * d.z));
            atomicAdd(&dL_dcov3Ds_inv[6 * global_id + 5], dL_dpower * (-0.5 * d.z * d.z));

            atomicAdd(&dL_ddensities[global_id], dL_daccum * exp_power);
        }
    }
}

void BACKWARD::preprocess(
    int P,
    const float* radii,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float* cov3Ds,
    const float* dL_dcov3Ds_inv,
    float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drotations
) {
    preprocessCUDA<<<(P + 255) / 256, 256>>>(
        P,
        radii,
        scales,
        rotations,
        cov3Ds,
        dL_dcov3Ds_inv,
        dL_dcov3Ds,
        dL_dscales,
        dL_drotations
    );
}

void BACKWARD::voxelize(
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
    const float* dL_daccum,
    float3* dL_dmeans3D,
    float* dL_ddensities,
    float* dL_dcov3Ds_inv 
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
        dL_daccum,
        dL_dmeans3D,
        dL_ddensities,
        dL_dcov3Ds_inv
    );
}
