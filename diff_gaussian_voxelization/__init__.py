from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def voxelize_gaussians(
    means3D,
    densities,
    scales,
    rotations,
    voxel_settings
):
    return _VoxelizeGaussians.apply(
        means3D,
        densities,
        scales,
        rotations,
        voxel_settings
    )

class _VoxelizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        means3D,
        densities,
        scales,
        rotations,
        voxel_settings
    ):
        args = (
            voxel_settings.voxel_physical,
            voxel_settings.volume_pixel,
            voxel_settings.volume_physical,
            means3D,
            densities,
            scales,
            rotations,
            voxel_settings.debug
        )
        num_voxelized, volume, radii, geomBuffer, binningBuffer, imgBuffer = _C.voxelize_gaussians(*args)

        ctx.voxel_settings = voxel_settings
        ctx.num_voxelized = num_voxelized
        ctx.save_for_backward(means3D, densities, scales, rotations, radii, geomBuffer, binningBuffer, imgBuffer)
        return volume, radii

    @staticmethod
    def backward(ctx, grad_volume, _):
        voxel_settings = ctx.voxel_settings
        num_voxelized = ctx.num_voxelized
        means3D, densities, scales, rotations, radii, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        args = (
            voxel_settings.voxel_physical,
            voxel_settings.volume_pixel,
            voxel_settings.volume_physical,
            means3D,
            densities,
            radii,
            scales,
            rotations,
            grad_volume,
            geomBuffer,
            num_voxelized,
            binningBuffer,
            imgBuffer,
            voxel_settings.debug
        )
        grad_densities, grad_means3D, grad_scales, grad_rotations = _C.voxelize_gaussians_backward(*args)
        
        grads = (
            grad_means3D,
            grad_densities,
            grad_scales,
            grad_rotations,
            None
        )

        return grads

class GaussianVoxelizationSettings(NamedTuple):
    voxel_physical: torch.Tensor
    volume_pixel: torch.Tensor
    volume_physical: torch.Tensor
    debug: bool

class GaussianVoxelizer(nn.Module):
    def __init__(self, voxel_settings):
        super().__init__()
        self.voxel_settings = voxel_settings

    def forward(self, means3D, densities, scales, rotations):
        voxel_settings = self.voxel_settings
        return voxelize_gaussians(
            means3D,
            densities,
            scales,
            rotations,
            voxel_settings
        )
