"""Camera presets and interactive visualization shared by preview and generation.

Poses are OpenCV camera-to-world matrices: +X right, +Y down, +Z forward.
"""
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go


TRAJECTORY_PRESETS = (
    "Spiral forward",
    "Dolly forward", "Dolly backward", "Truck left", "Truck right",
    "Pedestal up", "Pedestal down", "Pan left", "Pan right",
    "Tilt up", "Tilt down", "Orbit left", "Orbit right",
    "Crane forward", "S-curve", "Figure eight", "Static",
)


@dataclass
class CameraTrajectory:
    c2ws: np.ndarray
    intrinsics: np.ndarray
    keyframe_indices: np.ndarray
    preset: str
    resolution: int

    def to_dict(self):
        return {
            "preset": self.preset,
            "convention": "OpenCV camera-to-world: +X right, +Y down, +Z forward",
            "resolution": [self.resolution, self.resolution],
            "intrinsics_format": "fx, fy, cx, cy in pixels",
            "keyframe_indices": self.keyframe_indices.tolist(),
            "c2ws": self.c2ws.tolist(),
            "intrinsics": self.intrinsics.tolist(),
        }


def build_trajectory(preset="Spiral forward", translation_scale=1.0,
                     rotation_degrees=15.0, fov_degrees=60.0,
                     num_frames=120, resolution=256):
    """Build a dense path and its four conditioning cameras.

    A fixed set of four cameras determines the coordinate normalization, so
    changing video length does not change the scene's conditioning poses.
    Translation scale is applied AFTER normalization, preserving its effect.
    """
    if preset not in TRAJECTORY_PRESETS:
        raise ValueError(f"Unknown camera trajectory: {preset}")
    if not np.isfinite([translation_scale, rotation_degrees, fov_degrees]).all():
        raise ValueError("Camera settings must be finite.")
    if not 0 <= translation_scale <= 3 or not 0 <= rotation_degrees <= 60:
        raise ValueError("Translation scale must be in [0, 3] and rotation in [0, 60].")
    if not 20 <= fov_degrees <= 100:
        raise ValueError("Field of view must be between 20 and 100 degrees.")
    if int(num_frames) != num_frames or not 4 <= num_frames <= 600:
        raise ValueError("Video length must be an integer between 4 and 600 frames.")
    if resolution not in (256, 512):
        raise ValueError("Resolution must be 256 or 512.")
    num_frames = int(num_frames)
    # Uniform thirds would sample the same lateral position on every spiral
    # turn. Offset the two middle anchors to cover both sides of the helix.
    keyframe_times = (np.array([0, 2 / 9, 7 / 9, 1]) if preset == "Spiral forward"
                      else np.linspace(0, 1, 4))
    # Piecewise sampling preserves the exact conditioning poses across lengths.
    keyframes = np.rint(keyframe_times * (num_frames - 1)).astype(int)
    t = np.interp(np.arange(num_frames), keyframes, keyframe_times)
    theta = np.deg2rad(rotation_degrees) * t
    poses = np.repeat(np.eye(4)[None], num_frames, axis=0)
    positions = poses[:, :3, 3]
    yaw, pitch = np.zeros(num_frames), np.zeros(num_frames)
    if preset == "Spiral forward":
        # Three small helix turns around the forward axis, starting at the input pose.
        # Keep the camera looking forward; translation scale controls travel.
        phase = 6 * np.pi * t
        positions[:, 0] = 0.05 * (np.cos(phase) - 1)
        positions[:, 1] = 0.05 * np.sin(phase)
        positions[:, 2] = t
    elif preset in ("Dolly forward", "Dolly backward"):
        positions[:, 2] = t * (1 if preset == "Dolly forward" else -1)
    elif preset in ("Truck left", "Truck right"):
        positions[:, 0] = t * (-1 if preset == "Truck left" else 1)
    elif preset in ("Pedestal up", "Pedestal down"):
        positions[:, 1] = t * (-1 if preset == "Pedestal up" else 1)
    elif preset in ("Pan left", "Pan right"):
        yaw = theta * (-1 if preset == "Pan left" else 1)
    elif preset in ("Tilt up", "Tilt down"):
        pitch = theta * (1 if preset == "Tilt up" else -1)
    elif preset in ("Orbit left", "Orbit right"):
        arc = theta * (-1 if preset == "Orbit left" else 1)
        # Orbit a focus point at (0, 0, 1), starting from the input pose.
        positions[:, 0] = np.sin(arc)
        positions[:, 2] = 1 - np.cos(arc)
        yaw = -arc
    elif preset == "Crane forward":
        positions[:, 1], positions[:, 2] = -0.5 * t, t
        pitch = -theta / 2
    elif preset == "S-curve":
        positions[:, 0], positions[:, 2] = 0.35 * np.sin(2 * np.pi * t), t
        yaw = np.deg2rad(rotation_degrees) * np.sin(2 * np.pi * t)
    elif preset == "Figure eight":
        positions[:, 0] = np.sin(2 * np.pi * t)
        positions[:, 2] = 0.35 * np.sin(4 * np.pi * t)
        yaw = np.deg2rad(rotation_degrees) * np.sin(2 * np.pi * t)

    for i, (y, p) in enumerate(zip(yaw, pitch)):
        cy, sy, cp, sp = np.cos(y), np.sin(y), np.cos(p), np.sin(p)
        poses[i, :3, :3] = np.array([
            [cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]
        ])

    # Match RealEstate10K preprocessing: mean camera alignment and scale 1.35.
    anchors = poses[keyframes]
    forward = anchors[:, :3, 2].mean(0)
    forward /= np.linalg.norm(forward)
    right = np.cross(anchors[:, :3, 1].mean(0), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    average = np.eye(4)
    average[:3, :3] = np.stack([right, down, forward], axis=1)
    average[:3, 3] = anchors[:, :3, 3].mean(0)
    poses = np.linalg.inv(average) @ poses
    scale = 1.35 * np.abs(poses[keyframes, :3, 3]).max()
    if scale > 1e-8:
        poses[:, :3, 3] /= scale
    poses[:, :3, 3] *= translation_scale
    focal = resolution / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    intrinsics = np.tile([focal, focal, resolution / 2, resolution / 2], (num_frames, 1))
    return CameraTrajectory(poses.astype(np.float32), intrinsics.astype(np.float32),
                            keyframes, preset, resolution)


def trajectory_figure(trajectory):
    """Plot positions, oriented camera frusta, keyframes and travel arrows."""
    poses = trajectory.c2ws
    # Display with vertical up, while retaining OpenCV matrices for inference.
    display = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    centers = poses[:, :3, 3] @ display.T
    all_points = [centers]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0].tolist(), y=centers[:, 1].tolist(), z=centers[:, 2].tolist(),
        mode="lines+markers", name="Video path",
        line=dict(color="#4b8bff", width=5),
        marker=dict(size=2, color=list(range(len(poses))), colorscale="Viridis",
                    colorbar=dict(title="Frame")),
        text=[f"Frame {i}" for i in range(len(poses))], hoverinfo="text",
    ))
    keyframes = set(trajectory.keyframe_indices.tolist())
    samples = sorted(set(np.linspace(0, len(poses) - 1, 12, dtype=int)) | keyframes)
    for index in samples:
        pose = poses[index]
        fx, fy, cx, cy = trajectory.intrinsics[index]
        depth = 0.12
        corners = np.array([[0, 0, 0], [-cx / fx, -cy / fy, 1],
                            [(trajectory.resolution - cx) / fx, -cy / fy, 1],
                            [(trajectory.resolution - cx) / fx, (trajectory.resolution - cy) / fy, 1],
                            [-cx / fx, (trajectory.resolution - cy) / fy, 1]]) * depth
        world = (corners @ pose[:3, :3].T + pose[:3, 3]) @ display.T
        all_points.append(world)
        color = "#fb7185" if index == 0 else "#fbbf24" if index in keyframes else "#7c9ab8"
        points = []
        for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            points.extend([world[a].tolist(), world[b].tolist(), [None] * 3])
        coords = list(zip(*points))
        fig.add_trace(go.Scatter3d(
            x=list(coords[0]), y=list(coords[1]), z=list(coords[2]), mode="lines",
            line=dict(color=color, width=4 if index in keyframes else 2),
            name=f"{'Input' if index == 0 else 'Keyframe' if index in keyframes else 'Camera'} {index}",
            showlegend=index in keyframes,
        ))
        # Orientation arrows also distinguish pure rotation from a static path.
        origin = world[0]
        direction = pose[:3, 2] @ display.T
        fig.add_trace(go.Cone(
            x=[float(origin[0])], y=[float(origin[1])], z=[float(origin[2])],
            u=[float(direction[0])], v=[float(direction[1])], w=[float(direction[2])],
            sizemode="absolute", sizeref=0.055, anchor="tail",
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=False, hoverinfo="skip",
        ))
    arrow_indices = np.linspace(0, len(poses) - 2, 6, dtype=int)
    directions = centers[arrow_indices + 1] - centers[arrow_indices]
    lengths = np.linalg.norm(directions, axis=1)
    moving = lengths > 1e-8
    if moving.any():
        direction = directions[moving] / lengths[moving, None]
        points = centers[arrow_indices[moving]]
        fig.add_trace(go.Cone(
            x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist(),
            u=direction[:, 0].tolist(), v=direction[:, 1].tolist(), w=direction[:, 2].tolist(),
            sizemode="absolute", sizeref=0.07, anchor="tail",
            colorscale=[[0, "#34d399"], [1, "#34d399"]], showscale=False,
            name="Travel direction", hoverinfo="name",
        ))
    bounds = np.concatenate(all_points)
    midpoint = (bounds.max(0) + bounds.min(0)) / 2
    radius = max(float(np.ptp(bounds, axis=0).max()), 0.2) * 0.7
    # Equal axis ranges keep straight paths and pure-rotation frusta fully in
    # view; Plotly's data aspect ratio can clip a long, nearly flat trajectory.
    axes = [dict(title=label, range=[float(center - radius), float(center + radius)])
            for center, label in zip(midpoint, ["X (right)", "Z (forward)", "−Y (up)"])]
    fig.update_layout(
        title=trajectory.preset, height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis=axes[0], yaxis=axes[1], zaxis=axes[2], aspectmode="cube",
                   camera=dict(eye=dict(x=1.6, y=-1.8, z=1.2))),
        legend=dict(orientation="h", y=-0.04), uirevision="camera-preview",
    )
    return fig
