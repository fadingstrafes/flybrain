"""Camera presets and screen projection, independent of the live simulator."""

import math

import numpy as np


def camera_basis(yaw, pitch):
    outward = np.array([math.cos(pitch)*math.sin(yaw),
                        math.cos(pitch)*math.cos(yaw), math.sin(pitch)])
    right = np.array([-math.cos(yaw), math.sin(yaw), 0.0])
    up = np.cross(outward, right)
    return outward, right, up


def fit_camera(points, yaw=0.0, pitch=0.0, aspect=1.4, fov=42, padding=1.12):
    """Fit every supplied point including perspective depth, even in narrow windows."""
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    target = (low+high)*.5
    outward, right, up = camera_basis(yaw, pitch)
    delta = points-target
    vertical = math.tan(math.radians(fov)/2)
    needed = np.maximum(np.abs(delta @ right)/(vertical*max(aspect, .05)),
                        np.abs(delta @ up)/vertical)*padding + delta @ outward
    return {"yaw": yaw, "pitch": pitch, "distance": max(.2, float(needed.max())+.05),
            "target": target}


def arena_overview(arena, aspect=1.4):
    corners = [[x, y, z] for x in (-arena.half_width, arena.half_width)
               for y in (-arena.half_height, arena.half_height) for z in (0, 6)]
    return dict(fit_camera(corners, yaw=math.pi, pitch=1.40, aspect=aspect, fov=45),
                follow=False, preset="overview")


def pan_camera(camera, dx, dy, viewport_height, fov=42):
    _, right, up = camera_basis(camera["yaw"], camera["pitch"])
    scale = 2*camera["distance"]*math.tan(math.radians(fov)/2)/max(viewport_height, 1)
    camera["target"] = np.asarray(camera.get("target", np.zeros(3))) + (-dx*right+dy*up)*scale


def project_points(points, mvp, viewport):
    clip = np.column_stack([points, np.ones(len(points))]) @ mvp.T
    ndc = clip[:, :3]/np.where(np.abs(clip[:, 3:]) > 1e-9, clip[:, 3:], 1e-9)
    screen = np.column_stack([viewport[0]+(ndc[:, 0]+1)*viewport[2]/2,
                              viewport[1]+(ndc[:, 1]+1)*viewport[3]/2])
    visible = (clip[:, 3] > 0) & (np.abs(ndc) <= 1).all(axis=1)
    return screen, visible
