#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansCUDA(
    const torch::Tensor& voxel_physical_size,
    const torch::Tensor& volume_pixel_size,
    const torch::Tensor& volume_physical_size,
	const torch::Tensor& means3D,
    const torch::Tensor& densities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const bool debug
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansBackwardCUDA( 
    const torch::Tensor& voxel_physical_size,
    const torch::Tensor& volume_pixel_size,
    const torch::Tensor& volume_physical_size,
    const torch::Tensor& means3D,
    const torch::Tensor& densities,
    const torch::Tensor& radii,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& dL_daccum,
    const torch::Tensor& geomBuffer,
    const int num_voxelized,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug
);
