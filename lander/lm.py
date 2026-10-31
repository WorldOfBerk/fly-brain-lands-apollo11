"""Toy P66 model (phase 2): vertical dynamics only, B landings in parallel.

In P66 the commander clicks the ROD (rate of descent) switch; each click changes the commanded sink
rate by 1 ft/s (0.3048 m/s) and the AGC throttles to hold it. Here the fly does the clicking.

Approximate Apollo 11 values:
  P66 entry ~150 m (~500 ft), mass ~7.8 t, ~950 kg propellant
  DPS: 45.04 kN max, throttleable 10-60 %, Isp 311 s
  1.7 m probes trigger "contact light"; the crew shuts the engine down ~1 s later
  Landing gear limit: 10 ft/s (3.05 m/s) vertical
"""
from dataclasses import dataclass

import numpy as np

G_MOON = 1.62
G0 = 9.80665
FT = 0.3048


@dataclass
class LMParams:
    h0: float = 150.0  # m
    v0: float = -4.0  # m/s, negative = down
    dry_mass: float = 6850.0  # kg, without propellant
    fuel0: float = 950.0  # kg
    thrust_max: float = 45040.0  # N
    throttle_min: float = 0.10
    throttle_max: float = 0.60
    isp: float = 311.0  # s
    rod_tau: float = 1.5  # s, AGC P66 rate loop time constant
    throttle_lag: float = 0.3  # s, engine lag
    probe_len: float = 1.7  # m
    shutdown_delay: float = 1.0  # s, contact light -> shutdown
    gear_limit: float = 10 * FT  # m/s
    dt: float = 0.01  # s, physics step


class LunarModule:
    def __init__(self, n: int, p: LMParams | None = None):
        self.p = p or LMParams()
        self.n = n
        self.h = np.full(n, self.p.h0)
        self.v = np.full(n, self.p.v0)
        self.fuel = np.full(n, self.p.fuel0)
        self.thrust = np.full(n, (self.p.dry_mass + self.p.fuel0) * G_MOON)
        self.v_cmd = np.full(n, self.p.v0)
        self.engine_on = np.ones(n, bool)
        self.contact_t = np.full(n, np.nan)
        self.contact_v = np.full(n, np.nan)  # sink rate at contact light
        self.landed = np.zeros(n, bool)
        self.touchdown_v = np.full(n, np.nan)
        self.t = 0.0

    @property
    def mass(self):
        return self.p.dry_mass + self.fuel

    def rod_click(self, clicks: np.ndarray):
        """+1: slower (ROD up), -1: faster (ROD down). 1 ft/s per click."""
        self.v_cmd = np.minimum(self.v_cmd + clicks * FT, 0.0)

    def advance(self, duration: float):
        p = self.p
        for _ in range(round(duration / p.dt)):
            flying = ~self.landed
            # AGC P66: thrust to hold the commanded rate, within the throttle range
            a_cmd = (self.v_cmd - self.v) / p.rod_tau
            target = self.mass * (G_MOON + a_cmd)
            target = np.clip(target, p.throttle_min * p.thrust_max, p.throttle_max * p.thrust_max)
            target = np.where(self.engine_on & (self.fuel > 0), target, 0.0)
            self.thrust += (target - self.thrust) * (p.dt / p.throttle_lag)
            self.thrust = np.where(self.engine_on & (self.fuel > 0), self.thrust, 0.0)

            acc = self.thrust / self.mass - G_MOON
            self.v = np.where(flying, self.v + acc * p.dt, 0.0)
            self.h = np.where(flying, self.h + self.v * p.dt, self.h)
            self.fuel = np.where(flying, np.maximum(self.fuel - self.thrust / (p.isp * G0) * p.dt, 0.0), self.fuel)

            self.t += p.dt
            new_contact = flying & np.isnan(self.contact_t) & (self.h <= p.probe_len)
            self.contact_t[new_contact] = self.t
            self.contact_v[new_contact] = -self.v[new_contact]
            self.engine_on &= ~(self.t - self.contact_t >= p.shutdown_delay)

            down = flying & (self.h <= 0)
            self.touchdown_v[down] = -self.v[down]
            self.h[down] = 0.0
            self.landed |= down

    def outcome(self):
        ok = self.landed & (self.touchdown_v <= self.p.gear_limit)
        return np.where(~self.landed, "airborne", np.where(ok, "success", "crash"))
