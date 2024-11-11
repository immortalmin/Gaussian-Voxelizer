#include "voxelizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "helper_math.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__global__ void duplicateWithKeys(
	int P,
	const float* points_xyz,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float* radii,
	dim3 grid,
    const float3 voxel_physical,
    const float3 volume_offset,
    const float3 volume_center)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint3 rect_min, rect_max;
        float3 point_xyz = {points_xyz[3 * idx], points_xyz[3 * idx + 1], points_xyz[3 * idx + 2]};
		getRect(point_xyz, radii[idx], rect_min, rect_max, grid, voxel_physical, volume_offset, volume_center);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
        for (int z = rect_min.z; z < rect_max.z; ++z) {
    		for (int y = rect_min.y; y < rect_max.y; ++y)
    		{
    			for (int x = rect_min.x; x < rect_max.x; ++x)
    			{
                    uint32_t key = z * grid.y * grid.x + y * grid.x + x;
    				gaussian_keys_unsorted[off] = key;
    				gaussian_values_unsorted[off] = idx;
	    			off++;
		    	}
	    	}
        }
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currtile = point_list_keys[idx];
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1];
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

CudaVoxelizer::GeometryState CudaVoxelizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	//obtain(chunk, geom.depths, P, 128);
	//obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.cov3Ds, P * 6, 128);
	obtain(chunk, geom.cov3Ds_inv, P * 6, 128);
	//obtain(chunk, geom.conic_density, P, 128);
	//obtain(chunk, geom.coefs, P, 128);
	obtain(chunk, geom.blocks_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.blocks_touched, geom.blocks_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaVoxelizer::ImageState CudaVoxelizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	//obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaVoxelizer::BinningState CudaVoxelizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

int CudaVoxelizer::Voxelizer::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
    int P,
    const float* voxel_physical_size,
    const float* volume_pixel_size,
    const float* volume_center_pos,
    const float* means3D,
    const float* densities,
    const float* scales,
    const float* rotations,
    float* volume,
    float* radii,
    const bool debug
){
    // FIXME: 应该是float3还是float3*，因为后面要频繁地当作参数，可能使用引用或者指针更高效
    float3 voxel_physical = {voxel_physical_size[0], voxel_physical_size[1], voxel_physical_size[2]};
    float3 volume_pixel = {volume_pixel_size[0], volume_pixel_size[1], volume_pixel_size[2]};
    float3 volume_physical = voxel_physical * volume_pixel;
    float3 volume_center = {volume_center_pos[0], volume_center_pos[1], volume_center_pos[2]};
    float3 volume_offset = volume_physical / make_float3(2.0, 2.0, 2.0);

    dim3 block_grid((volume_pixel.x + BLOCK_X - 1) / BLOCK_X, (volume_pixel.y + BLOCK_Y - 1) / BLOCK_Y, (volume_pixel.z + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if(radii == nullptr) {
        radii = geomState.internal_radii;
    }

    // Dynamically resize volume-based auxiliary buffers during training
    // FIXME: change the name of the variable
	size_t img_chunk_size = required<ImageState>(block_grid.x * block_grid.y * block_grid.z);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, block_grid.x * block_grid.y * block_grid.z);

    CHECK_CUDA(FORWARD::preprocess(
        P,
        means3D,
        densities,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        geomState.cov3Ds,
        geomState.cov3Ds_inv,
        radii,
        block_grid,
        voxel_physical,
        volume_offset,
        volume_center,
        geomState.blocks_touched
    ), debug)
    
    cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.blocks_touched, geomState.point_offsets, P);

    int num_voxelized;
    cudaMemcpy(&num_voxelized, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
    
    size_t binning_chunk_size = required<BinningState>(num_voxelized);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_voxelized);
    cudaMemset(binningState.point_list_keys_unsorted, 0, block_grid.x * block_grid.y * block_grid.z * sizeof(uint32_t));

    duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		block_grid,
        voxel_physical,
        volume_offset,
        volume_center
    );
    CHECK_CUDA(, debug)
    
    int bit = getHigherMsb(block_grid.x * block_grid.y * block_grid.z);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_voxelized, 0, bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, block_grid.x * block_grid.y * block_grid.z * sizeof(uint2)), debug)
    
	// Identify start and end of per-tile workloads in sorted list
	if (num_voxelized > 0)
		identifyTileRanges << <(num_voxelized + 255) / 256, 256 >> > (
			num_voxelized,
			binningState.point_list_keys,
			imgState.ranges);
    CHECK_CUDA(, debug)

    CHECK_CUDA(FORWARD::voxelize(
        block_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        voxel_physical,
        volume_pixel,
        volume_physical,
        volume_center,
        means3D,
        densities,
        geomState.cov3Ds_inv,
        volume
    ), debug)
    return num_voxelized;
}


void CudaVoxelizer::Voxelizer::backward(
    const int P, int R,
    const float* voxel_physical_size,
    const float* volume_pixel_size,
    const float* volume_center_pos,
    const float* means3D,
    const float* densities,
    const float* scales,
    const float* rotations,
    const float* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_daccums,
    float* dL_ddensities,
    float* dL_dmeans3D,
    float* dL_dcov3Ds,
    float* dL_dcov3Ds_inv,
    float* dL_dscales,
    float* dL_drotations,
    const bool debug
) {
    float3 voxel_physical = {voxel_physical_size[0], voxel_physical_size[1], voxel_physical_size[2]};
    float3 volume_pixel = {volume_pixel_size[0], volume_pixel_size[1], volume_pixel_size[2]};
    float3 volume_physical = voxel_physical * volume_pixel;
    float3 volume_center = {volume_center_pos[0], volume_center_pos[1], volume_center_pos[2]};

    dim3 block_grid((volume_pixel.x + BLOCK_X - 1) / BLOCK_X, (volume_pixel.y + BLOCK_Y - 1) / BLOCK_Y, (volume_pixel.z + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, block_grid.x * block_grid.y * block_grid.z);

    // FIXME: 既然geomState.internal_radii保存了radii，还有必要传radii么？
    if(radii == nullptr) {
        radii = geomState.internal_radii;
    }

    CHECK_CUDA(BACKWARD::voxelize(
        block_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        voxel_physical,
        volume_pixel,
        volume_physical,
        volume_center,
        means3D,
        densities,
        geomState.cov3Ds_inv,
        dL_daccums,
        (float3*)dL_dmeans3D,
        dL_ddensities,
        dL_dcov3Ds_inv
    ), debug)

    CHECK_CUDA(BACKWARD::preprocess(
        P,
        radii,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        geomState.cov3Ds,
        dL_dcov3Ds_inv,
        dL_dcov3Ds,
        (glm::vec3*)dL_dscales,
        (glm::vec4*)dL_drotations
    ), debug)
}
