"""Transfer Gaussian colors to extracted meshes and export portable previews."""
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import trimesh


PREVIEW_VERSION = 2


def read_gaussian_colors(point_cloud_path):
    """Read world positions, display RGB colors and Gaussian opacities."""
    points = PlyData.read(point_cloud_path)["vertex"].data
    if not len(points):
        raise ValueError("A nonempty Gaussian cloud is required.")
    xyz = np.column_stack([points[axis] for axis in ("x", "y", "z")])
    colors = np.column_stack([points[f"f_dc_{axis}"] for axis in range(3)])
    colors = np.clip(0.5 + 0.28209479177387814 * colors, 0, 1)
    if not np.isfinite(xyz).all() or not np.isfinite(colors).all():
        raise ValueError("Gaussian positions and colors must be finite.")
    opacity = np.ones(len(points), dtype=np.float64)
    if "opacity" in points.dtype.names:
        opacity = 1 / (1 + np.exp(-np.clip(points["opacity"].astype(np.float64), -80, 80)))
    if not np.isfinite(opacity).all():
        raise ValueError("Gaussian opacities must be finite.")
    return xyz, colors, opacity


def color_mesh_from_gaussians(mesh, point_cloud_path, neighbors=8):
    """Interpolate visible Gaussian DC colors at the mesh's world-space vertices."""
    if not len(mesh.vertices):
        raise ValueError("A nonempty mesh is required.")
    xyz, colors, opacity = read_gaussian_colors(point_cloud_path)
    count = min(max(1, int(neighbors)), len(xyz))
    distances, indices = cKDTree(xyz).query(np.asarray(mesh.vertices), k=count)
    distances = distances.reshape(len(mesh.vertices), count)
    indices = indices.reshape(len(mesh.vertices), count)
    weights = 1 / np.maximum(distances, 1e-6) ** 2
    weights *= opacity[indices]
    weights /= weights.sum(axis=1, keepdims=True)
    interpolated = np.sum(colors[indices] * weights[..., None], axis=1)
    rgba = np.column_stack([np.rint(interpolated * 255).astype(np.uint8),
                            np.full(len(mesh.vertices), 255, dtype=np.uint8)])
    colored = mesh.copy()
    colored.visual.vertex_colors = rgba
    return colored


def export_colored_mesh(mesh, point_cloud_path, obj_path, glb_path):
    """Save a textured Z-up OBJ package and self-contained, Y-up PBR GLB."""
    import zipfile
    from .mesh_texture import texture_mesh_from_gaussians

    textured = texture_mesh_from_gaussians(mesh, point_cloud_path)
    obj_path, glb_path = Path(obj_path), Path(glb_path)
    obj_data, assets = trimesh.exchange.obj.export_obj(
        textured, include_normals=True, include_texture=True, return_texture=True,
        mtl_name=obj_path.with_suffix(".mtl").name,
    )
    preview = textured.copy()
    preview.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    glb_data = trimesh.exchange.gltf.export_glb(trimesh.Scene(preview), include_normals=True)
    # Keep the OBJ, material and PNG together for a portable download.
    package = obj_path.with_suffix(".obj.zip")
    package_temp = package.with_name(package.name + ".tmp")
    with zipfile.ZipFile(package_temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(obj_path.name, obj_data)
        for name, data in assets.items():
            archive.writestr(name, data)
    for name, data in assets.items():
        target = obj_path.parent / name
        temp = target.with_name(target.name + ".tmp")
        temp.write_bytes(data if isinstance(data, bytes) else data.encode())
        temp.replace(target)
    obj_temp = obj_path.with_name(obj_path.name + ".tmp")
    glb_temp = glb_path.with_name(glb_path.name + ".tmp")
    obj_temp.write_text(obj_data)
    glb_temp.write_bytes(glb_data)
    obj_temp.replace(obj_path)
    glb_temp.replace(glb_path)
    package_temp.replace(package)
    return textured
