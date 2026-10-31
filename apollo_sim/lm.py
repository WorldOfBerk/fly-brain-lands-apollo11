"""LM physics: 2D orbital mechanics in a Moon-centred vertical plane, DPS engine, RCS attitude.

Moon-centred inertial plane with the landing site at (0, R_MOON).
`pitch` is the angle of the thrust axis from the landing-site vertical.
"""
from dataclasses import dataclass, field

import numpy as np

from . import constants as C


@dataclass
class LMState:
    pos: np.ndarray  # m, Moon-centred
    vel: np.ndarray  # m/s
    mass: float
    propellant: float
    pitch: float = 0.0  # rad, + = thrust tilted downrange
    pitch_rate: float = 0.0
    thrust: float = 0.0  # N, actual (lagged) thrust
    t: float = 0.0
    landed: bool = False
    touchdown_v: tuple = field(default=(np.nan, np.nan))  # (vertical, horizontal) m/s


class LunarModule:
    def __init__(self, state: LMState, terrain=None):
        self.s = state
        self.thrust_cmd = 0.0  # N
        self.pitch_cmd = 0.0  # rad
        self.engine_on = True
        self.terrain = terrain  # apollo_sim.terrain.Terrain or None (flat Moon)
        self.touchdown_slope = np.nan  # deg

    @property
    def datum_altitude(self) -> float:
        """Height above the mean lunar radius."""
        return float(np.linalg.norm(self.s.pos) - C.R_MOON)

    @property
    def ground_height(self) -> float:
        return float(self.terrain.height(self.s.pos[0])) if self.terrain is not None else 0.0

    @property
    def altitude(self) -> float:
        """Height above the terrain under the footpads."""
        return self.datum_altitude - self.ground_height

    def local_frame(self):
        """Local (up, downrange) unit vectors."""
        up = self.s.pos / np.linalg.norm(self.s.pos)
        down_range = np.array([up[1], -up[0]])  # orbit is clockwise
        return up, down_range

    def vertical_speed(self) -> float:
        up, _ = self.local_frame()
        return float(self.s.vel @ up)

    def horizontal_speed(self) -> float:
        _, dr = self.local_frame()
        return float(self.s.vel @ dr)

    def step(self, dt: float):
        s = self.s
        if s.landed:
            return
        # attitude: rate-limited PD with RCS torque limit
        inertia = C.PITCH_INERTIA_PER_KG * s.mass
        alpha_max = C.RCS_PITCH_TORQUE / inertia
        rate_cmd = np.clip(1.5 * (self.pitch_cmd - s.pitch), -C.ATT_MAX_RATE, C.ATT_MAX_RATE)
        alpha = np.clip((rate_cmd - s.pitch_rate) / 0.3, -alpha_max, alpha_max)
        s.pitch_rate += alpha * dt
        s.pitch += s.pitch_rate * dt

        # DPS: first-order lag (THROTLAG), stops when out of propellant
        cmd = self.thrust_cmd if (self.engine_on and s.propellant > C.UNUSABLE_PROPELLANT * 0) else 0.0
        s.thrust += (cmd - s.thrust) * min(1.0, dt / C.THROTLAG)
        if not self.engine_on or s.propellant <= 0:
            s.thrust = 0.0

        # thrust direction in the landing-site guidance frame
        site_up = np.array([0.0, 1.0])
        site_dr = np.array([1.0, 0.0])
        thrust_dir = np.cos(s.pitch) * site_up + np.sin(s.pitch) * site_dr
        r = np.linalg.norm(s.pos)
        acc = -C.MU_MOON * s.pos / r**3 + thrust_dir * s.thrust / s.mass

        s.vel = s.vel + acc * dt
        s.pos = s.pos + s.vel * dt
        mdot = s.thrust / (C.DPS_ISP * C.G0)
        s.propellant = max(0.0, s.propellant - mdot * dt)
        s.mass = C.MASS_PDI - (C.PROPELLANT_PDI - s.propellant)
        s.t += dt

        if self.altitude <= 0.0:
            up, dr = self.local_frame()
            s.touchdown_v = (float(-s.vel @ up), float(s.vel @ dr))
            if self.terrain is not None:
                self.touchdown_slope = self.terrain.slope_under(s.pos[0])
            s.landed = True
            s.vel = np.zeros(2)


def at_ignition(terrain=None) -> LunarModule:
    """State at PDI ignition (t = 0).

    RIGN/VIGN in the pad load describe the state at throttle-up, after the 26 s trim at 10 % thrust,
    so the state is integrated 26 s backwards at 10 % thrust to get the ignition point.
    """
    x, z = C.RIGN
    pos = np.array([z, C.R_MOON + x])  # landing site at (0, R), downrange along +x
    up = pos / np.linalg.norm(pos)
    dr = np.array([up[1], -up[0]])
    vel = dr * C.VIGN
    thrust = C.DPS_THROTTLE_MIN * C.DPS_NOMINAL_MAX
    mdot = thrust / (C.DPS_ISP * C.G0)
    mass = C.MASS_PDI - mdot * C.ZOOMTIME
    dt = 0.05
    for _ in range(round(C.ZOOMTIME / dt)):  # backwards
        r = np.linalg.norm(pos)
        acc = -C.MU_MOON * pos / r**3 + np.array([-1.0, 0.0]) * thrust / mass
        vel = vel - acc * dt
        pos = pos - vel * dt
        mass += mdot * dt
    st = LMState(pos=pos, vel=vel, mass=C.MASS_PDI, propellant=C.PROPELLANT_PDI, pitch=-np.pi / 2)
    return LunarModule(st, terrain)
