"""Binocular scene sampling and an explicit engineered collision-avoidance assist.

RGB and depth are ray-cast from the two eye positions. Depth-assisted threat
estimation and its mapping to visual relay neurons are modeling assumptions.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class VisualFrame:
    images: np.ndarray
    depths: np.ndarray
    clearances: np.ndarray  # left, forward, right; eye-to-obstacle distances
    signals: np.ndarray  # two eyes, nine horizontal sectors
    looming: float


class BinocularVision:
    width, height, sectors = 90, 30, 9
    max_range = 12.0

    def __init__(self):
        horizontal = np.linspace(math.radians(75), math.radians(-75), self.width)
        vertical = np.linspace(math.radians(30), math.radians(-45), self.height)
        az, el = np.meshgrid(horizontal, vertical)
        self.azimuth = np.stack([az + math.radians(35), az - math.radians(35)])
        self.elevation = np.stack([el, el])
        self.previous_depth = None

    def reset(self):
        self.previous_depth = None

    def sample(self, arena, dt):
        shape = self.azimuth.shape
        azimuth = self.azimuth + arena.yaw
        direction = np.stack([-np.sin(azimuth)*np.cos(self.elevation),
                              np.cos(azimuth)*np.cos(self.elevation), np.sin(self.elevation)], axis=-1)
        origins = []
        for side in (-1, 1):
            xy = arena.position + arena.rotation() @ np.array([side*.20, .60])
            xy = np.clip(xy, [-arena.half_width+.01, -arena.half_height+.01],
                         [arena.half_width-.01, arena.half_height-.01])
            origins.append([*xy, arena.altitude+.58])
        origin = np.broadcast_to(np.asarray(origins)[:,None,None,:], (*shape, 3))
        depth = np.full(shape, self.max_range)
        kind = np.zeros(shape, dtype=np.int8)
        color = np.broadcast_to(np.array([27., 43., 60.]), (*shape, 3)).copy()

        def surface(distance, tint, label, allowed=True):
            visible = (distance > 0) & (distance < depth) & allowed
            depth[visible] = distance[visible]
            kind[visible] = label
            color[visible] = tint

        for axis, bounds in [(0, (-arena.half_width, arena.half_width)),
                             (1, (-arena.half_height, arena.half_height)), (2, (0, 6))]:
            for bound in bounds:
                distance = np.divide(bound-origin[...,axis], direction[...,axis],
                                     out=np.full(shape, np.inf), where=np.abs(direction[...,axis]) > 1e-8)
                # The room has a floor, four walls and a visible ceiling.
                tint = [45, 68, 74] if axis == 2 and bound == 0 else [90, 111, 124]
                surface(distance, tint, 1 if axis == 2 else 2)
        for index, (x, y, radius) in enumerate(arena.obstacles):
            # Eye origins lie less than one unit from the body. Skip distant
            # objects before allocating any ray/ellipsoid intersection arrays.
            if (abs(x-arena.position[0]) > self.max_range+radius+1
                    or abs(y-arena.position[1]) > self.max_range+radius+1):
                continue
            scales = np.array([radius, radius, .55])
            offset = (origin-[x,y,.32])/scales
            ray = direction/scales
            a = np.sum(ray*ray, axis=-1)
            b = np.sum(offset*ray, axis=-1)
            c = np.sum(offset*offset, axis=-1)-1
            discriminant = b*b-a*c
            distance = (-b-np.sqrt(np.maximum(0, discriminant)))/a
            surface(distance, [92+index%3*19, 126, 117-index%3*12], 3, discriminant >= 0)
        point = origin + direction*depth[...,None]
        # High-contrast grid/stripe detail makes orientation visible in the eye feeds.
        grid = (np.mod(point[...,0], 1) < .045) | (np.mod(point[...,1], 1) < .045)
        color[(kind == 1) & grid] *= 1.7
        stripes = (np.mod(point[...,0]+point[...,1], 2) < .15)
        color[(kind == 2) & stripes] *= .55
        color *= (.45 + .55*np.exp(-depth/18))[...,None]
        hazard = (kind >= 2) & (point[...,2] >= arena.altitude-.08) & (point[...,2] <= arena.altitude+.9)
        hazards = np.where(hazard, depth, self.max_range)
        ranges = []
        for low, high in [(25, 100), (-25, 25), (-100, -25)]:
            mask = (np.degrees(self.azimuth) >= low) & (np.degrees(self.azimuth) <= high)
            ranges.append(float(hazards[mask].min()))
        closing = np.zeros(shape)
        if self.previous_depth is not None:
            closing = np.clip((self.previous_depth-depth)/max(dt, 1e-6), 0, 8)/8
        self.previous_depth = depth.copy()
        proximity = np.clip(1-hazards/self.max_range, 0, 1)
        brightness = color.mean(axis=-1)/255
        signals = np.zeros((2, self.sectors), dtype=np.float32)
        for eye in range(2):
            for sector, columns in enumerate(np.array_split(np.arange(self.width), self.sectors)):
                signals[eye,sector] = np.clip(.7*proximity[eye][:,columns].max()
                    + .2*closing[eye][:,columns].mean() + .1*brightness[eye][:,columns].std(), 0, 1)
        return VisualFrame(np.clip(color,0,255).astype(np.uint8), depth.astype(np.float32),
                           np.asarray(ranges, dtype=np.float32), signals, float(closing[hazard].mean()) if hazard.any() else 0)


class VisualNavigator:
    """Reflex assist over perceived clearances; never reads position or obstacle coordinates."""
    def __init__(self):
        self.direction = 1.0
        self.turning = False
        self.active = False
        self.interventions = 0

    def command(self, clearances, speed):
        left, front, right = map(float, clearances)
        lookahead = 2.4 + 1.0*speed
        was_active = self.active
        if not self.turning and front < lookahead:
            self.turning = True
            if abs(left-right) > .15:
                self.direction = 1.0 if left > right else -1.0
        if self.turning and front > lookahead+1.0:
            self.turning = False
        self.active = self.turning
        if self.active and not was_active:
            self.interventions += 1
        if self.active:
            throttle = float(np.clip((front-.9)/(lookahead+.2), 0, .55))
            return self.direction*2.8, throttle
        # A gentle lateral correction prevents skimming walls with the head.
        side = np.clip((right-left)*.25, -1, 1) if min(left,right) < 1.4 else 0
        return -float(side), 1.0
