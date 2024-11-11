#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_voxelizer/config.h"
#include "cuda_voxelizer/voxelizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansCUDA(
    const torch::Tensor& voxel_physical_size,
    const torch::Tensor& volume_pixel_size,
    const torch::Tensor& volume_center_pos,
	const torch::Tensor& means3D,
    const torch::Tensor& densities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const bool debug
) {
    const int P = means3D.size(0);

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    
    //FIXME:这里轴的顺序可能会错。现在三个轴一样大，所以没报错。
	torch::Tensor volume = torch::full({volume_pixel_size[0].item().to<int>(), volume_pixel_size[1].item().to<int>(), volume_pixel_size[2].item().to<int>()}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0.0, float_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int voxelized = 0;
    if(P != 0) {
    	voxelized = CudaVoxelizer::Voxelizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P,
            voxel_physical_size.contiguous().data<float>(),
            volume_pixel_size.contiguous().data<float>(),
            volume_center_pos.contiguous().data<float>(),
    		means3D.contiguous().data<float>(), 
    		densities.contiguous().data<float>(), 
    		scales.contiguous().data_ptr<float>(), 
    		rotations.contiguous().data_ptr<float>(),
    		volume.contiguous().data<float>(),
            radii.contiguous().data<float>(),
            debug
	    );
    }
	return std::make_tuple(voxelized, volume, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
VoxelizeGaussiansBackwardCUDA(
    const torch::Tensor& voxel_physical_size,
    const torch::Tensor& volume_pixel_size,
    const torch::Tensor& volume_center_pos,
    const torch::Tensor& means3D,
    const torch::Tensor& densities,
    const torch::Tensor& radii,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& dL_daccum,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug
) {
    const int P = means3D.size(0);

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_ddensities = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    torch::Tensor dL_dcov3Ds = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dcov3Ds_inv = torch::zeros({P, 6}, means3D.options());

    if(P != 0) {
        CudaVoxelizer::Voxelizer::backward(
            P, R,
            voxel_physical_size.contiguous().data<float>(),
            volume_pixel_size.contiguous().data<float>(),
            volume_center_pos.contiguous().data<float>(),
            means3D.contiguous().data<float>(),
            densities.contiguous().data<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            radii.contiguous().data<float>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_daccum.contiguous().data<float>(),
            dL_ddensities.contiguous().data<float>(),
            dL_dmeans3D.contiguous().data<float>(),
            dL_dcov3Ds.contiguous().data<float>(),
            dL_dcov3Ds_inv.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_drotations.contiguous().data<float>(),
            debug
        );   
    }

    return std::make_tuple(dL_ddensities, dL_dmeans3D, dL_dscales, dL_drotations);

}


