## Overview
This library is an implementation of Gaussian Voxelizer, which is similar to [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian).
What r2_gaussian is doing is applying [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) in the field of CBCT, as can be seen in their [repository](https://github.com/Ruyi-Zha/r2_gaussian).

There are some differences between our voxelizer and theirs:
1. They first convert the 3D Gaussians to a voxel space, and then take the volume. And we directly extract the volume from the world space.
2. We currently do not have the flexibility of r2_gaussian, which can extract volumes from any position. We will make improvements to this in the future. (Finished ✅)
3. But I think our code executes faster, and we will add comparative experiments in the future.

The overall structure of the project and some codes are referenced from 3D-GS.
And this project was completed before r2_gaussian was open sourced.
However, the part that calculates the Gaussian radius is based on the reference r2_gaussian (at that time, I didn't know that the radius could be calculated directly based on the scale)
## Installation
```sh
git clone https://github.com/immortalmin/Gaussian-Voxelizer.git

cd Gaussian-Voxelizer
pip install .
```

## Usage
```Python
from diff_gaussian_voxelization import GaussianVoxelizationSettings, GaussianVoxelizer

voxel_settings = GaussianVoxelizationSettings(
    voxel_physical=voxel_physical,      # the physical size of a voxel. e.g. torch.tensor([0.001, 0.001, 0.001])
    volume_pixel=volume_pixel,          # the pixel size of the volume. e.g. torch.tensor([128, 128, 128])
    volume_physical=volume_physical,    # the physical size of the volume. e.g. torch.tensor([0.128, 0.128, 0.128])
    debug=True
)

voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)
# res_cuda is a 3D matrix representing the volume
res_cuda, radii_cuda = voxelizer(
    self.gaussians.get_xyz,
    self.gaussians.get_density,
    self.gaussians.get_scaling,
    self.gaussians.get_rotation
)
```
