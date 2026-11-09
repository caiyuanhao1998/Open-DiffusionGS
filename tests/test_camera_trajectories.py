import json
import unittest

import numpy as np

from camera_trajectories import TRAJECTORY_PRESETS, build_trajectory, trajectory_figure


class CameraTrajectoryTests(unittest.TestCase):
    def test_spiral_is_default_and_existing_presets_keep_their_order(self):
        self.assertEqual(TRAJECTORY_PRESETS[0], "Spiral forward")
        self.assertEqual(build_trajectory().preset, "Spiral forward")
        self.assertEqual(TRAJECTORY_PRESETS[1:], (
            "Dolly forward", "Dolly backward", "Truck left", "Truck right",
            "Pedestal up", "Pedestal down", "Pan left", "Pan right",
            "Tilt up", "Tilt down", "Orbit left", "Orbit right",
            "Crane forward", "S-curve", "Figure eight", "Static",
        ))

    def test_spiral_advances_along_a_circle_with_forward_facing_cameras(self):
        path = build_trajectory("Spiral forward", num_frames=61)
        relative = np.linalg.inv(path.c2ws[0]) @ path.c2ws
        positions = relative[:, :3, 3]
        self.assertTrue((np.diff(positions[:, 2]) > 0).all())
        # A full helix turn returns to its lateral start, but ends farther ahead.
        np.testing.assert_allclose(positions[-1, :2], positions[0, :2], atol=1e-6)
        radius = 0.25 * positions[-1, 2]
        radial_distance = np.linalg.norm(positions[:, :2] + [radius, 0], axis=1)
        np.testing.assert_allclose(radial_distance, radius, atol=1e-6)
        self.assertGreater(np.ptp(positions[:, 1]), radius)
        np.testing.assert_allclose(relative[:, :3, :3],
                                   np.tile(np.eye(3), (61, 1, 1)), atol=1e-6)

    def test_all_presets_are_valid_opencv_cameras(self):
        for name in TRAJECTORY_PRESETS:
            with self.subTest(preset=name):
                path = build_trajectory(name)
                self.assertEqual(path.c2ws.shape, (60, 4, 4))
                self.assertTrue(np.isfinite(path.c2ws).all())
                rotations = path.c2ws[:, :3, :3]
                np.testing.assert_allclose(rotations.transpose(0, 2, 1) @ rotations,
                                           np.broadcast_to(np.eye(3), rotations.shape), atol=1e-6)
                np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
                np.testing.assert_allclose(path.c2ws[:, 3], np.tile([0, 0, 0, 1], (60, 1)))
                self.assertEqual(len(np.unique(path.keyframe_indices)), 4)
                figure = trajectory_figure(path)
                self.assertGreater(len(figure.data), 4)
                exported = json.loads(json.dumps(path.to_dict()))
                np.testing.assert_array_equal(exported["keyframe_indices"], path.keyframe_indices)
                np.testing.assert_allclose(exported["c2ws"], path.c2ws)

    def test_semantic_translation_directions(self):
        for name, axis, sign in [("Truck right", 0, 1), ("Truck left", 0, -1),
                                 ("Pedestal up", 1, -1), ("Pedestal down", 1, 1),
                                 ("Dolly forward", 2, 1), ("Dolly backward", 2, -1)]:
            with self.subTest(preset=name):
                p = build_trajectory(name).c2ws
                relative = np.linalg.inv(p[0]) @ p[-1]
                self.assertGreater(sign * relative[axis, 3], 0)

    def test_rotation_presets_do_not_translate(self):
        for name, axis, sign in [("Pan right", 0, 1), ("Pan left", 0, -1),
                                 ("Tilt up", 1, -1), ("Tilt down", 1, 1)]:
            with self.subTest(preset=name):
                p = build_trajectory(name).c2ws
                np.testing.assert_allclose(p[:, :3, 3], 0, atol=1e-6)
                relative = np.linalg.inv(p[0]) @ p[-1]
                self.assertGreater(sign * relative[axis, 2], 0)

    def test_translation_scale_is_not_cancelled_by_normalization(self):
        a = build_trajectory("Dolly forward", translation_scale=0.5)
        b = build_trajectory("Dolly forward", translation_scale=1)
        np.testing.assert_allclose(a.c2ws[:, :3, 3] * 2, b.c2ws[:, :3, 3])

    def test_video_length_does_not_change_model_keyframes(self):
        for name in TRAJECTORY_PRESETS:
            with self.subTest(preset=name):
                a = build_trajectory(name, num_frames=12)
                b = build_trajectory(name, num_frames=61)
                np.testing.assert_allclose(a.c2ws[a.keyframe_indices], b.c2ws[b.keyframe_indices], atol=1e-6)

    def test_resolution_only_scales_intrinsics(self):
        a = build_trajectory(resolution=256)
        b = build_trajectory(resolution=512)
        np.testing.assert_array_equal(a.c2ws, b.c2ws)
        np.testing.assert_allclose(a.intrinsics * 2, b.intrinsics)

    def test_closed_and_static_paths(self):
        p = build_trajectory("Figure eight").c2ws
        np.testing.assert_allclose(p[0], p[-1], atol=1e-6)
        p = build_trajectory("Static").c2ws
        np.testing.assert_allclose(p, np.tile(np.eye(4), (60, 1, 1)), atol=1e-6)

    def test_degenerate_motion_remains_finite(self):
        for name in TRAJECTORY_PRESETS:
            with self.subTest(preset=name):
                p = build_trajectory(name, translation_scale=0, rotation_degrees=0)
                self.assertTrue(np.isfinite(p.c2ws).all())

    def test_invalid_camera_settings_are_rejected(self):
        for options in [dict(preset="unknown"), dict(fov_degrees=0),
                        dict(rotation_degrees=float("nan")), dict(num_frames=3),
                        dict(num_frames=12.5), dict(resolution=1024)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_trajectory(**options)


if __name__ == "__main__":
    unittest.main()
