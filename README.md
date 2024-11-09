This is an implementation of the Gaussian Voxelizer

## Installation
```sh
git clone https://github.com/immortalmin/Gaussian-Voxelizer.git

cd Gaussian-Voxelizer
pip install .
```

## Usage
```Python
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
