#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED
#include <vector>
#include <functional>

namespace CudaVoxelizer {
    class Voxelizer {
        public:
            static int forward(
                std::function<char* (size_t)> geometryBuffer,
                std::function<char* (size_t)> binningBuffer,
			    std::function<char* (size_t)> imageBuffer,
                int P,
                const float* voxel_physical_size,
                const float* volume_pixel_size,
                const float* volume_physical_size,
                const float* means3D,
                const float* densities,
                const float* scales,
                const float* rotations,
                float* volume,
                float* radii,
                const bool debug
            );
            
            static void backward(
                const int P, int R,
                const float* voxel_physical_size,
                const float* volume_pixel_size,
                const float* volume_physical_size,
                const float* means3D,
                const float* densities,
                const float* scales,
                const float* rotations,
                const float* radii,
                char* geom_buffer,
                char* binning_buffer,
                char* image_buffer,
                const float* dL_daccum,
                float* dL_ddensities,
                float* dL_dmeans3D,
                float* dL_dcov3Ds,
                float* dL_dcov3Ds_inv,
                float* dL_dscales,
                float* dL_drotations,
                const bool debug
            );

    };
};

#endif
