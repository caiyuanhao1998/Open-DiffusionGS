"""Transfer Gaussian colors to extracted meshes and export portable previews."""
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import trimesh


PREVIEW_VERSION = 1


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


def _baked_color_material(tree):
    # Gaussian colors already contain the appearance predicted by the model.
    # Unlit glTF materials display those colors consistently without depending
    # on the viewer having an environment map or additional scene lights.
    tree["materials"] = [{
        "name": "Gaussian colors",
        "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                 "metallicFactor": 0, "roughnessFactor": 1},
        "doubleSided": True,
        "extensions": {"KHR_materials_unlit": {}},
    }]
    extensions = tree.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in extensions:
        extensions.append("KHR_materials_unlit")
    for item in tree["meshes"]:
        for primitive in item["primitives"]:
            primitive["material"] = 0


def export_colored_mesh(mesh, point_cloud_path, obj_path, glb_path):
    """Save a colored Z-up OBJ and a self-contained, Y-up GLB for the viewer."""
    colored = color_mesh_from_gaussians(mesh, point_cloud_path)
    obj_path, glb_path = Path(obj_path), Path(glb_path)
    obj_data = trimesh.exchange.obj.export_obj(colored, include_normals=True, include_color=True)
    preview = colored.copy()
    preview.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    # glTF vertex colors are linear; the model's display RGB values are sRGB.
    rgb = preview.visual.vertex_colors[:, :3].astype(np.float64) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    preview.visual.vertex_colors = np.column_stack([
        np.rint(linear * 255).astype(np.uint8), np.full(len(rgb), 255, dtype=np.uint8),
    ])
    glb_data = trimesh.exchange.gltf.export_glb(
        trimesh.Scene(preview), include_normals=True, tree_postprocessor=_baked_color_material,
    )
    # Publish complete files, including when recoloring previously extracted meshes.
    obj_temp = obj_path.with_name(obj_path.name + ".tmp")
    glb_temp = glb_path.with_name(glb_path.name + ".tmp")
    obj_temp.write_text(obj_data)
    glb_temp.write_bytes(glb_data)
    obj_temp.replace(obj_path)
    glb_temp.replace(glb_path)
    return colored
