"""Bake Gaussian appearance into UV textures, as used by the project-page meshes."""
import numpy as np
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import trimesh
import xatlas


@njit(cache=True)
def _raster_positions(vertices, faces, uv, resolution):
    positions = np.zeros((resolution, resolution, 3), np.float32)
    mask = np.zeros((resolution, resolution), np.uint8)
    for face in faces:
        ia, ib, ic = face
        ax, ay = uv[ia, 0] * resolution, (1 - uv[ia, 1]) * resolution
        bx, by = uv[ib, 0] * resolution, (1 - uv[ib, 1]) * resolution
        cx, cy = uv[ic, 0] * resolution, (1 - uv[ic, 1]) * resolution
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1e-12:
            continue
        for y in range(max(0, int(np.floor(min(ay, by, cy)))),
                       min(resolution - 1, int(np.ceil(max(ay, by, cy)))) + 1):
            for x in range(max(0, int(np.floor(min(ax, bx, cx)))),
                           min(resolution - 1, int(np.ceil(max(ax, bx, cx)))) + 1):
                a = ((by - cy) * (x + .5 - cx) + (cx - bx) * (y + .5 - cy)) / denominator
                b = ((cy - ay) * (x + .5 - cx) + (ax - cx) * (y + .5 - cy)) / denominator
                c = 1 - a - b
                if a >= -1e-6 and b >= -1e-6 and c >= -1e-6:
                    for axis in range(3):
                        positions[y, x, axis] = (a * vertices[ia, axis] + b * vertices[ib, axis]
                                                + c * vertices[ic, axis])
                    mask[y, x] = 1
    return positions, mask


def texture_mesh_from_gaussians(mesh, point_cloud_path, resolution=2048):
    """Return a Z-up mesh with smooth normals and an embedded sRGB color texture."""
    from .mesh_preview import read_gaussian_colors

    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("A finite, nonempty mesh is required.")
    surface = mesh.copy()
    surface.fix_normals(multibody=True)
    # Keep the original geometry separately so cached exports never accumulate smoothing.
    if len(surface.faces) > 100:
        trimesh.smoothing.filter_taubin(surface, lamb=.5, nu=.53, iterations=8)
    vertices = np.asarray(surface.vertices, dtype=np.float32)
    faces = np.asarray(surface.faces, dtype=np.uint32)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart = xatlas.ChartOptions()
    chart.max_iterations = 1
    pack = xatlas.PackOptions()
    pack.resolution, pack.padding, pack.bilinear = resolution, 4, True
    atlas.generate(chart, pack)
    mapping, uv_faces, uv = atlas[0]
    mapped = vertices[mapping]
    positions, mask = _raster_positions(mapped, uv_faces.astype(np.int64), uv, resolution)
    occupied = mask.astype(bool)
    if not occupied.any():
        raise ValueError("Mesh UV rasterization produced no surface texels.")
    queries = positions[occupied]
    del positions
    xyz, colors, opacity = read_gaussian_colors(point_cloud_path)
    usable = opacity > .02
    if usable.sum() > 100:
        xyz, colors, opacity = xyz[usable], colors[usable], opacity[usable]
    tree = cKDTree(xyz)
    paint = np.empty((len(queries), 3), np.uint8)
    neighbors = min(12, len(xyz))
    for start in range(0, len(queries), 65536):
        query = queries[start:start + 65536]
        distances, indices = tree.query(query, k=neighbors, workers=2)
        distances, indices = distances.reshape(len(query), neighbors), indices.reshape(len(query), neighbors)
        weights = opacity[indices] / np.maximum(distances, 1e-6) ** 2
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
        rgb = (colors[indices] * weights[..., None]).sum(axis=1)
        paint[start:start + len(query)] = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    image = np.zeros((resolution, resolution, 3), np.uint8)
    image[occupied] = paint
    # Extend island colors into padding to avoid dark seams under mipmapping.
    nearest = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    image[~occupied] = image[nearest[0, ~occupied], nearest[1, ~occupied]]
    material = trimesh.visual.material.PBRMaterial(
        name="Gaussian appearance", baseColorTexture=Image.fromarray(image),
        baseColorFactor=[1., 1., 1., 1.], metallicFactor=0., roughnessFactor=1., doubleSided=True,
    )
    return trimesh.Trimesh(
        vertices=mapped, faces=uv_faces, vertex_normals=np.asarray(surface.vertex_normals)[mapping],
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False,
    )
