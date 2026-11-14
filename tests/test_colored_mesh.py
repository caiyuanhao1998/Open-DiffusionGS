"""Colored mesh export must retain geometry and work with cached demo results."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement
import trimesh

from diffusionGS.utils.mesh_preview import color_mesh_from_gaussians, export_colored_mesh
from object_demo import ObjectDemo


def write_cloud(path, xyz, rgb, opacity=None):
    fields = [(name, "f4") for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")]
    data = np.zeros(len(xyz), dtype=fields)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = np.asarray(xyz)[:, axis]
        data[f"f_dc_{axis}"] = (np.asarray(rgb)[:, axis] - 0.5) / 0.28209479177387814
    data["opacity"] = 10 if opacity is None else opacity
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


class ColoredMeshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                                    faces=[[0, 1, 2]], process=False)
        self.rgb = np.eye(3)
        write_cloud(self.root / "test.ply", self.mesh.vertices, self.rgb)

    def test_color_transfer_preserves_geometry_and_matches_surface_colors(self):
        colored = color_mesh_from_gaussians(self.mesh, self.root / "test.ply")
        np.testing.assert_array_equal(colored.vertices, self.mesh.vertices)
        np.testing.assert_array_equal(colored.faces, self.mesh.faces)
        np.testing.assert_array_equal(colored.visual.vertex_colors[:, :3], self.rgb * 255)

    def test_low_opacity_outlier_does_not_dominate_surface_color(self):
        xyz = np.vstack([self.mesh.vertices, self.mesh.vertices[0]])
        rgb = np.vstack([self.rgb, [0, 0, 1]])
        write_cloud(self.root / "test.ply", xyz, rgb, [10, 10, 10, -10])
        colored = color_mesh_from_gaussians(self.mesh, self.root / "test.ply")
        self.assertGreater(colored.visual.vertex_colors[0, 0], 250)
        self.assertLess(colored.visual.vertex_colors[0, 2], 5)

    def test_glb_contains_colors_normals_and_self_contained_material(self):
        obj, glb = self.root / "test.obj", self.root / "mesh_preview.glb"
        export_colored_mesh(self.mesh, self.root / "test.ply", obj, glb)
        data = glb.read_bytes()
        size, kind = struct.unpack_from("<I4s", data, 12)
        self.assertEqual(kind, b"JSON")
        tree = json.loads(data[20:20 + size])
        attributes = tree["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("TEXCOORD_0", attributes)
        self.assertIn("NORMAL", attributes)
        material = tree["materials"][0]
        self.assertNotIn("KHR_materials_unlit", material.get("extensions", {}))
        self.assertEqual(material["pbrMetallicRoughness"]["metallicFactor"], 0)
        self.assertEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 1)
        self.assertIn("baseColorTexture", material["pbrMetallicRoughness"])
        self.assertTrue(all("uri" not in image for image in tree["images"]))
        self.assertTrue(all("uri" not in buffer for buffer in tree["buffers"]))
        exported = trimesh.load(obj, force="mesh", process=False)
        np.testing.assert_allclose(exported.bounds, self.mesh.bounds)
        self.assertEqual(exported.visual.kind, "texture")
        preview = trimesh.load(glb, force="mesh", process=False)
        self.assertEqual(preview.visual.material.baseColorTexture.size, (2048, 2048))
        with zipfile.ZipFile(obj.with_suffix(".obj.zip")) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("test.obj", archive.namelist())
            self.assertIn("test.mtl", archive.namelist())
            material = archive.read("test.mtl").decode()
            texture = next(line.split(maxsplit=1)[1] for line in material.splitlines()
                           if line.startswith("map_Kd "))
            self.assertIn(texture, archive.namelist())
        self.assertIn("\nvn ", obj.read_text())

    def test_existing_uncolored_demo_result_can_be_upgraded_without_a_gpu(self):
        self.mesh.export(self.root / "test.obj")
        (self.root / "result.json").write_text(json.dumps({"mesh_exported": True}))
        service = ObjectDemo(output_dir=self.root)
        preview, obj, glb, status = service.extract_mesh(str(self.root))
        self.assertEqual(preview, glb)
        self.assertTrue(zipfile.is_zipfile(obj))
        self.assertTrue((self.root / "mesh_geometry.ply").is_file())
        self.assertTrue(Path(preview).is_file())
        self.assertTrue(json.loads((self.root / "result.json").read_text())["colored_mesh"])
        modified = Path(glb).stat().st_mtime_ns
        service.extract_mesh(str(self.root))
        self.assertEqual(Path(glb).stat().st_mtime_ns, modified)


    def test_geometry_callback_finishes_before_texture_export_and_is_cached(self):
        from unittest.mock import patch
        from diffusionGS.utils.mesh_preview import export_colored_mesh as real_export
        (self.root / "result.json").write_text(json.dumps({"mesh_exported": False}))
        service = ObjectDemo(output_dir=self.root)
        active, calls = [], []

        def gpu_geometry(directory):
            active.append(True)
            calls.append(directory)
            self.mesh.export(self.root / "test.obj")
            active.pop()

        def cpu_export(*args, **kwargs):
            self.assertFalse(active)
            return real_export(*args, **kwargs)

        service.geometry_extractor = gpu_geometry
        with patch("diffusionGS.utils.mesh_preview.export_colored_mesh", side_effect=cpu_export):
            result = service.extract_mesh(str(self.root))
            service.extract_mesh(str(self.root))
        self.assertEqual(calls, [str(self.root)])
        self.assertTrue(Path(result[0]).is_file())
        self.assertTrue(zipfile.is_zipfile(result[1]))


if __name__ == "__main__":
    unittest.main()
