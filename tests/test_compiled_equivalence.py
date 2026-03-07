"""Equivalence tests: original vs Compiled (torch.compile-friendly) API.

Compares outputs and gradients of fully_fused_projection vs fully_fused_projection_compiled,
rasterize_to_pixels vs rasterize_to_pixels_compiled, and spherical_harmonics vs
spherical_harmonics_compiled.

Input conventions (aligned with test_rasterization.py and test_basic.py):
- Projection and rasterize tests use load_test_data() for realistic geometry and camera:
  Ks (3x3 intrinsics), viewmats (4x4 world-to-camera), width/height, means, quats
  (unit-norm), scales (small positive), opacities in [0, 1]. Only colors/backgrounds
  are randomized in [0, 1] where needed.
- Spherical harmonics test uses unit-norm dirs (view directions), random SH coeffs,
  and optional boolean masks.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from typing_extensions import Tuple

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def _expand(data: dict, batch_dims: Tuple[int, ...]):
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            ret[k] = v.expand(batch_dims + v.shape)
        else:
            ret[k] = v
    return ret


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (2,)])
def test_fully_fused_projection_compiled_equivalence(
    test_data, calc_compensations: bool, batch_dims: Tuple[int, ...]
):
    """Call fully_fused_projection twice (two clone sets); assert outputs and grads match.
    Swap the second call to fully_fused_projection_compiled to test compiled equivalence.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        fully_fused_projection_compiled,
        quat_scale_to_covar_preci,
    )

    torch.manual_seed(42)
    test_data = _expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]

    # Two separate clone sets (same values) so each path has independent tensors
    means = test_data["means"].clone().detach().requires_grad_(True)
    quats = test_data["quats"].clone().detach().requires_grad_(True)
    scales = test_data["scales"].clone().detach().requires_grad_(True)
    viewmats = test_data["viewmats"].clone().detach().requires_grad_(True)

    means2 = test_data["means"].clone().detach().requires_grad_(True)
    quats2 = test_data["quats"].clone().detach().requires_grad_(True)
    scales2 = test_data["scales"].clone().detach().requires_grad_(True)
    viewmats2 = test_data["viewmats"].clone().detach().requires_grad_(True)

    # Verify both paths get identical inputs (ensures test setup is correct)
    torch.testing.assert_close(means, means2, rtol=0, atol=0)
    torch.testing.assert_close(quats, quats2, rtol=0, atol=0)
    torch.testing.assert_close(scales, scales2, rtol=0, atol=0)
    torch.testing.assert_close(viewmats, viewmats2, rtol=0, atol=0)

    # Path 1: uncompiled
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        packed=False,
        calc_compensations=calc_compensations,
        camera_model="pinhole",
    )

    # Path 2: compiled (torch.compile the compiled API)
    proj_compiled = torch.compile(fully_fused_projection_compiled, fullgraph=True)
    radii2, means2d2, depths2, conics2, compensations2 = proj_compiled(
        means2,
        None,
        quats2,
        scales2,
        viewmats2,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
        camera_model="pinhole",
    )

    # Compare forward outputs before any backward().
    # The kernel does not write means2d/depths/conics/compensations when it early-returns
    # (radii set to 0); those elements are undefined. Only compare at valid positions.
    torch.testing.assert_close(radii, radii2, rtol=0, atol=0)
    valid = (radii[..., 0] > 0) & (radii[..., 1] > 0)
    n_valid = valid.sum().item()
    assert n_valid > 0, (
        "test data produced no valid projections (radii > 0); "
        "ensure scene/camera setup yields visible Gaussians"
    )
    assert n_valid >= min(100, valid.numel() // 10), (
        f"too few valid projections ({n_valid}); need meaningful coverage of projection path"
    )
    # Compare only at valid indices (radii non-zero => kernel wrote these outputs)
    valid_2 = valid.unsqueeze(-1).expand_as(means2d)
    torch.testing.assert_close(
        means2d[valid_2],
        means2d2[valid_2],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        depths[valid], depths2[valid], rtol=1e-5, atol=1e-5
    )
    valid_3 = valid.unsqueeze(-1).expand_as(conics)
    torch.testing.assert_close(
        conics[valid_3],
        conics2[valid_3],
        rtol=1e-5,
        atol=1e-5,
    )
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid],
            compensations2[valid],
            rtol=1e-5,
            atol=1e-5,
        )

    # Backward on both paths, then compare grads
    loss_orig = (
        means2d.sum() + depths.sum() + conics.sum()
        + (compensations.sum() if compensations is not None else 0.0)
    )
    loss_orig.backward()
    g_means_orig = means.grad.clone()
    g_quats_orig = quats.grad.clone()
    g_scales_orig = scales.grad.clone()
    g_viewmats_orig = viewmats.grad.clone()

    loss_comp = (
        means2d2.sum() + depths2.sum() + conics2.sum()
        + (compensations2.sum() if compensations2 is not None else 0.0)
    )
    loss_comp.backward()

    torch.testing.assert_close(g_means_orig, means2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(g_quats_orig, quats2.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_scales_orig, scales2.grad, rtol=1e-3, atol=1e-3)
    # viewmats grad is accumulated via atomics in backward; order is non-deterministic
    torch.testing.assert_close(
        g_viewmats_orig, viewmats2.grad, rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("with_backgrounds", [True, False])
@pytest.mark.parametrize("batch_dims", [()])
def test_rasterize_to_pixels_compiled_equivalence(
    test_data, with_backgrounds: bool, batch_dims: Tuple[int, ...]
):
    """Call rasterize_to_pixels twice (two clone sets); assert outputs and grads match.
    Swap the second call to rasterize_to_pixels_compiled to test compiled equivalence."""
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
        rasterize_to_pixels_compiled,
    )

    torch.manual_seed(42)
    channels = 3
    test_data = _expand(test_data, batch_dims)
    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    I = math.prod(batch_dims) * C

    # Geometry and camera from load_test_data (realistic Ks, viewmats, width, height)
    means = test_data["means"].clone().detach()
    quats = test_data["quats"].clone().detach()
    scales = test_data["scales"].clone().detach() * 0.1  # small scales for stable projection
    viewmats = test_data["viewmats"]
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]
    opacities = test_data["opacities"]
    opacities = torch.broadcast_to(
        opacities[..., None, :], batch_dims + (C, N)
    ).clone().detach()

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )

    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # Colors and backgrounds in [0, 1] (realistic RGB) - two clone sets
    colors = torch.rand(batch_dims + (C, N, channels), device=device).detach()
    backgrounds = (
        torch.rand(batch_dims + (C, channels), device=device).detach()
        if with_backgrounds
        else None
    )

    means2d = means2d.detach().requires_grad_(True)
    conics = conics.detach().requires_grad_(True)
    colors = colors.detach().requires_grad_(True)
    opacities = opacities.detach().requires_grad_(True)
    if backgrounds is not None:
        backgrounds = backgrounds.detach().requires_grad_(True)

    means2d2 = means2d.detach().clone().requires_grad_(True)
    conics2 = conics.detach().clone().requires_grad_(True)
    colors2 = colors.detach().clone().requires_grad_(True)
    opacities2 = opacities.detach().clone().requires_grad_(True)
    backgrounds2 = (
        backgrounds.detach().clone().requires_grad_(True)
        if backgrounds is not None
        else None
    )

    # Path 1: uncompiled
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    # Path 2: compiled (torch.compile the compiled API)
    rasterize_compiled = torch.compile(rasterize_to_pixels_compiled, fullgraph=True)
    render_colors2, render_alphas2 = rasterize_compiled(
        means2d2,
        conics2,
        colors2,
        opacities2,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds2,
    )

    # Compare forward outputs before any backward()
    torch.testing.assert_close(render_colors, render_colors2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(render_alphas, render_alphas2, rtol=1e-5, atol=1e-5)

    # Backward on both paths, then compare grads
    loss_orig = render_colors.sum() + render_alphas.sum()
    loss_orig.backward()
    g_means2d_orig = means2d.grad.clone()
    g_conics_orig = conics.grad.clone()
    g_colors_orig = colors.grad.clone()
    g_opacities_orig = opacities.grad.clone()
    g_backgrounds_orig = backgrounds.grad.clone() if backgrounds is not None else None

    loss_comp = render_colors2.sum() + render_alphas2.sum()
    loss_comp.backward()

    torch.testing.assert_close(g_means2d_orig, means2d2.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(g_conics_orig, conics2.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_colors_orig, colors2.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_opacities_orig, opacities2.grad, rtol=8e-3, atol=6e-3)
    if with_backgrounds:
        torch.testing.assert_close(
            g_backgrounds_orig, backgrounds2.grad, rtol=1e-3, atol=1e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 2])
@pytest.mark.parametrize("with_masks", [True, False])
def test_spherical_harmonics_compiled_equivalence(
    sh_degree: int, with_masks: bool
):
    """Call spherical_harmonics twice (two clone sets); assert outputs and grads match.
    Swap the second call to spherical_harmonics_compiled to test compiled equivalence."""
    from gsplat.cuda._wrapper import spherical_harmonics, spherical_harmonics_compiled

    torch.manual_seed(42)
    N = 200
    K = (sh_degree + 1) ** 2
    # Unit-norm view directions (realistic; matches usage in rendering) - two clone sets
    dirs = F.normalize(torch.randn(N, 3, device=device), dim=-1).detach().requires_grad_(True)
    coeffs = torch.randn(N, K, 3, device=device).detach().requires_grad_(True)
    masks = (
        torch.rand(N, device=device) > 0.3
        if with_masks
        else None
    )
    if masks is not None:
        masks = masks.to(device)

    dirs2 = dirs.detach().clone().requires_grad_(True)
    coeffs2 = coeffs.detach().clone().requires_grad_(True)
    masks2 = masks.clone() if masks is not None else None

    # Path 1: uncompiled
    colors = spherical_harmonics(sh_degree, dirs, coeffs, masks=masks)

    # Path 2: compiled (torch.compile the compiled API)
    sh_compiled = torch.compile(spherical_harmonics_compiled, fullgraph=True)
    colors2 = sh_compiled(sh_degree, dirs2, coeffs2, masks=masks2)

    # Compare forward outputs before any backward()
    torch.testing.assert_close(colors, colors2, rtol=1e-5, atol=1e-5)

    # Backward on both paths, then compare grads
    loss_orig = colors.sum()
    loss_orig.backward()
    g_dirs_orig = dirs.grad.clone()
    g_coeffs_orig = coeffs.grad.clone()

    loss_comp = colors2.sum()
    loss_comp.backward()

    torch.testing.assert_close(g_dirs_orig, dirs2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(g_coeffs_orig, coeffs2.grad, rtol=1e-4, atol=1e-4)
