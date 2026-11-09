"""Extracted meshes must retain the Gaussian cloud's coordinate system."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusionGS.models.gsrenderer.gs_core import GaussianModel, gaussian_3d_coeff


class MeshCoordinateTests(unittest.TestCase):
    def test_spatially_indexed_field_matches_dense_tile_reference(self):
        generator = torch.Generator().manual_seed(17)
        model = GaussianModel(sh_degree=0)
        model._xyz = torch.rand(9, 3, generator=generator) * 2 - 1
        model._scaling = torch.full((9, 3), np.log(0.08))
        model._rotation = torch.randn(9, 4, generator=generator)
        model._opacity = torch.ones(9, 1)
        resolution, blocks = 16, 4
        for relax_ratio in (0, 1.5, 2.25):
            with self.subTest(relax_ratio=relax_ratio), patch("kiui.lo"):
                actual = model.extract_fields(resolution, blocks, relax_ratio)
                centers = (model.get_xyz - model.mesh_center) * model.mesh_scale
                covariance = model.covariance_activation(
                    model.get_scaling * model.mesh_scale, 1, model._rotation,
                )
                grid = torch.linspace(-1, 1, resolution)
                xyz = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
                starts = torch.arange(resolution) // (resolution // blocks) * (resolution // blocks)
                lower = grid[starts] - (2 / blocks) * relax_ratio
                upper = grid[starts + resolution // blocks - 1] + (2 / blocks) * relax_ratio
                lows = torch.stack(torch.meshgrid(lower, lower, lower, indexing="ij"), dim=-1)
                highs = torch.stack(torch.meshgrid(upper, upper, upper, indexing="ij"), dim=-1)
                selected = ((centers > lows[..., None, :]) & (centers < highs[..., None, :])).all(-1)
                offsets = xyz[..., None, :] - centers
                covariances = covariance.expand(*offsets.shape[:-1], 6)
                weights = gaussian_3d_coeff(offsets.reshape(-1, 3), covariances.reshape(-1, 6))
                weights = weights.reshape(*xyz.shape[:-1], len(centers))
                expected = (weights * selected * model.get_opacity.flatten()).sum(-1)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_mesh_restores_translation_and_scale_after_field_extraction(self):
        model = GaussianModel.__new__(GaussianModel)
        model.mesh_center = torch.tensor([2.0, -3.0, 4.0])
        model.mesh_scale = 2.0
        grid = torch.linspace(-1, 1, 32)
        xyz = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
        field = (xyz.square().sum(-1) < 0.5 ** 2).float()
        model.extract_fields = lambda *args, **kwargs: field
        with patch("diffusionGS.models.gsrenderer.gs_core.clean_mesh", side_effect=lambda v, f, **kw: (v, f)):
            mesh = model.extract_mesh(density_thresh=0.5, resolution=32, decimate_target=0)
        np.testing.assert_allclose(mesh.bounds.mean(0), model.mesh_center.numpy(), atol=1e-5)
        np.testing.assert_allclose(mesh.extents, 0.5, atol=0.05)


if __name__ == "__main__":
    unittest.main()
