"""Deterministic arena/body kinematics. All lengths and dynamics are toy units."""

from dataclasses import dataclass, field
import math

import numpy as np


DEFAULT_ARENA_SIZE = (480.0, 360.0)
START_OBSTACLES = ((-2.3, 1.5, .85), (2.5, -.3, 1.0), (-9, 7, 1.3),
                   (8, 8, 1.2), (-12, -9, 1.4), (13, -8, 1.1), (0, 12, 1.5))


def arena_obstacles(half_width, half_height):
    """Deterministic sparse landmarks; keep the familiar starting neighborhood.

    Grid spacing grows for very large custom rooms to bound simulation/render
    cost. Radii and fly size do not grow with the room.
    """
    result = [o for o in START_OBSTACLES
              if abs(o[0])+o[2]+1 < half_width and abs(o[1])+o[2]+1 < half_height]
    spacing = max(28.0, max(half_width, half_height)/10)
    rng = np.random.default_rng(731)
    for x in np.arange(-half_width+spacing/2, half_width, spacing):
        for y in np.arange(-half_height+spacing/2, half_height, spacing):
            px, py = np.array([x, y]) + rng.uniform(-spacing*.2, spacing*.2, 2)
            radius = float(rng.uniform(.8, 2.2))
            if abs(px) < 20 and abs(py) < 17:
                continue
            if abs(px)+radius+1 < half_width and abs(py)+radius+1 < half_height:
                result.append((float(px), float(py), radius))
    return tuple(result)


def leg_hip(leg):
    return np.array([(-1 if leg < 3 else 1) * .22, [.38, .12, -.1][leg % 3], .44])


def solve_knee(hip, foot, bend_hint, upper=.55, lower=.60):
    """Two-segment inverse kinematics with a chosen outward bending plane."""
    delta = np.asarray(foot) - hip
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    distance = np.clip(distance, abs(upper - lower) + 1e-5, upper + lower - 1e-5)
    along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
    height = math.sqrt(max(0, upper * upper - along * along))
    bend = np.asarray(bend_hint) - direction * np.dot(bend_hint, direction)
    if np.linalg.norm(bend) < 1e-6:
        bend = np.cross(direction, [0, 1, 0])
    bend /= max(1e-6, np.linalg.norm(bend))
    return hip + direction * along + bend * height


class LegGait:
    """World-space planted feet and timed swing arcs; no neural gait claim."""
    def __init__(self):
        self.feet = np.zeros((6, 3))
        self.starts = np.zeros((6, 3))
        self.targets = np.zeros((6, 3))
        self.progress = np.ones(6)
        self.phase_was_swing = np.zeros(6, dtype=bool)
        self.airborne = False

    @staticmethod
    def rest_foot(leg):
        return np.array([(-1 if leg < 3 else 1) * .86, [.91, -.02, -.84][leg % 3], .02])

    def reset(self, position, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c,-s,0], [s,c,0], [0,0,1]])
        origin = np.array([*position, 0.0])
        self.feet = np.array([origin + R @ self.rest_foot(i) for i in range(6)])
        self.progress.fill(1)
        self.phase_was_swing.fill(False)
        self.airborne = False

    def update(self, position, yaw, altitude, speed, phase, dt):
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c,-s,0], [s,c,0], [0,0,1]])
        origin = np.array([*position, altitude])
        if altitude > .08:
            fold = min(1, altitude / .55)
            for leg in range(6):
                rest = self.rest_foot(leg)
                tucked = np.array([(-1 if leg < 3 else 1)*.27, [.3, .05, -.2][leg % 3], .21])
                self.feet[leg] = origin + R @ (rest * (1-fold) + tucked * fold)
            self.airborne = True
            return
        if self.airborne:
            self.reset(position, yaw)
        for leg in range(6):
            local_phase = (phase / (2 * math.pi) + (.5 if leg in (1,3,5) else 0)) % 1
            swing_phase = local_phase > .62
            hip = origin + R @ leg_hip(leg)
            stretched = np.linalg.norm(self.feet[leg] - hip) > 1.08
            if self.progress[leg] >= 1 and (stretched or (swing_phase and not self.phase_was_swing[leg] and speed > .04)):
                self.starts[leg] = self.feet[leg]
                self.targets[leg] = np.array([*position, 0.0]) + R @ (self.rest_foot(leg) + [0, .16, 0])
                self.progress[leg] = 0
            self.phase_was_swing[leg] = swing_phase
            if self.progress[leg] < 1:
                self.progress[leg] = min(1, self.progress[leg] + dt / (.16 + .08 * (1 - min(1, speed))))
                p = self.progress[leg]
                smooth = p*p*(3-2*p)
                self.feet[leg] = self.starts[leg] * (1-smooth) + self.targets[leg] * smooth
                self.feet[leg, 2] += .16 * math.sin(math.pi * p)**2


@dataclass
class Arena:
    half_width: float = DEFAULT_ARENA_SIZE[0]/2
    half_height: float = DEFAULT_ARENA_SIZE[1]/2
    radius: float = 0.48
    obstacles: tuple | None = None
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, -3.5]))
    yaw: float = 0.0
    speed: float = 0.0
    turn_rate: float = 0.0
    phase: float = 0.0
    elapsed: float = 0.0
    distance: float = 0.0
    contacts: int = 0
    altitude: float = 0.0
    vertical_speed: float = 0.0
    gait: LegGait = field(default_factory=LegGait)

    def __post_init__(self):
        if not np.isfinite([self.half_width, self.half_height]).all() or min(self.half_width, self.half_height) < 6:
            raise ValueError("Arena half dimensions must be finite and at least 6")
        if self.obstacles is None:
            self.obstacles = arena_obstacles(self.half_width, self.half_height)
        self.gait.reset(self.position, self.yaw)

    def nearby_obstacles(self, distance):
        """Broad phase shared by touch, collision and finite-range eye rays."""
        return tuple(o for o in self.obstacles
                     if abs(o[0]-self.position[0]) <= distance+o[2]
                     and abs(o[1]-self.position[1]) <= distance+o[2])

    def reset(self):
        self.position = np.array([0.0, -3.5])
        self.yaw = self.speed = self.turn_rate = self.phase = 0.0
        self.elapsed = self.distance = 0.0
        self.contacts = 0
        self.altitude = self.vertical_speed = 0.0
        self.gait.reset(self.position, self.yaw)

    def rotation(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def sense(self):
        """Near-contact probes and stride-phase feedback to annotated leg sensors."""
        touch = np.zeros(6, dtype=np.float32)
        proprio = np.zeros(6, dtype=np.float32)
        nearby = self.nearby_obstacles(2)
        for leg in range(6):
            side = -1 if leg < 3 else 1
            pair = leg % 3
            probe = self.position + self.rotation() @ np.array([side * 0.8, [0.9, 0, -0.8][pair]])
            clearance = min(self.half_width - abs(probe[0]), self.half_height - abs(probe[1]))
            for x, y, radius in nearby:
                clearance = min(clearance, np.linalg.norm(probe - [x, y]) - radius)
            touch[leg] = np.clip(1 - clearance / 0.6, 0, 1) if self.altitude < .65 else 0
            offset = math.pi if leg in (1, 3, 5) else 0
            proprio[leg] = min(1, self.speed / 1.0) * (0.5 + 0.5 * math.sin(self.phase + offset))
        return touch, proprio

    def step(self, activation, dt, wing_activation=(0, 0), flight_requested=False,
             navigation=None):
        if dt <= 0 or dt > 0.1:
            raise ValueError("Arena steps must be in (0, 0.1] seconds")
        activation = np.asarray(activation, dtype=np.float64)
        if activation.shape != (6,) or not np.isfinite(activation).all():
            raise ValueError("Expected six finite motor activation levels")
        # Smooth saturation retains steering differences between strongly active sides.
        levels = 1.0 - np.exp(-3.0 * np.clip(activation, 0, 1))
        left, right = float(levels[:3].mean()), float(levels[3:].mean())
        target_speed = 2.0 * (left + right) * 0.5
        target_turn = 2.5 * (right - left)
        wing = np.clip(np.asarray(wing_activation, dtype=float), 0, 1)
        if wing.shape != (2,) or not np.isfinite(wing).all():
            raise ValueError("Expected two finite wing activation levels")
        power = float(wing.mean())
        # Lift needs measured wing motor output. A flight request only changes
        # target altitude; it cannot lift a fly whose wing motors are silent.
        target_altitude = 1.8 if flight_requested else 0.0
        desired_lift = max(0, 4.0 + 4*(target_altitude-self.altitude) - 3*self.vertical_speed)
        lift = min(12*power, desired_lift)
        self.vertical_speed += (lift - 4.0) * dt
        self.altitude = float(np.clip(self.altitude + self.vertical_speed * dt, 0, 3.0))
        if self.altitude == 0 and self.vertical_speed < 0:
            self.vertical_speed = 0
        if self.altitude >= 3 and self.vertical_speed > 0:
            self.vertical_speed = 0
        if self.altitude > .08:
            target_speed = 2.6 * power
            target_turn = 3.0 * float(wing[1] - wing[0])
        if navigation is not None:
            steering, throttle = navigation
            # Explicit visual assist in the kinematic decoder, not an emergent
            # connectome reflex. It cannot power a body with no motor activity.
            motor_power = power if self.altitude > .08 else (left+right)*.5
            authority = min(1.0, motor_power*5)
            target_speed *= float(np.clip(throttle, 0, 1))
            if abs(steering) > .01:
                target_turn = float(steering)*authority
        alpha = 1 - math.exp(-dt / 0.15)
        self.speed += (target_speed - self.speed) * alpha
        self.turn_rate += (target_turn - self.turn_rate) * alpha
        self.yaw = (self.yaw + self.turn_rate * dt + math.pi) % (2 * math.pi) - math.pi
        before = self.position.copy()
        proposed = before + np.array([-math.sin(self.yaw), math.cos(self.yaw)]) * self.speed * dt
        self.position = np.clip(proposed, [-self.half_width + self.radius, -self.half_height + self.radius],
                                [self.half_width - self.radius, self.half_height - self.radius])
        hit = not np.allclose(proposed, self.position, atol=1e-10, rtol=0)
        for x, y, radius in self.nearby_obstacles(self.radius):
            if self.altitude > .9:
                continue
            offset = self.position - [x, y]
            distance = np.linalg.norm(offset)
            minimum = radius + self.radius
            if distance < minimum:
                normal = offset / distance if distance > 1e-10 else np.array([1.0, 0.0])
                self.position = np.array([x, y]) + normal * minimum
                hit = True
        self.contacts += int(hit)
        travelled = float(np.linalg.norm(self.position - before))
        self.distance += travelled
        # The gait is a visual oscillator driven by actual displacement.
        self.phase = (self.phase + travelled * 8.0) % (2 * math.pi)
        self.elapsed += dt
        self.gait.update(self.position, self.yaw, self.altitude, self.speed, self.phase, dt)

    def pose(self):
        return {"position": self.position.copy(), "yaw": self.yaw, "speed": self.speed,
                "phase": self.phase, "elapsed": self.elapsed, "distance": self.distance,
                "contacts": self.contacts, "altitude": self.altitude,
                "vertical_speed": self.vertical_speed, "feet": self.gait.feet.copy()}
