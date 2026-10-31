"""The fly's eye: a compound eye looking down from the LM, photoreceptors and motion detection.

Pipeline:
  1) Scene: apollo_sim.terrain.Terrain (ray cast), or LunarSurface below, a flat plane with crater
     shading under the 10.8 deg Apollo 11 sun.
  2) Eye: ommatidia on a 5 deg azimuth x elevation grid (Drosophila ~4.5-5 deg), each seeing luminance
     band-limited to its own footprint. Sky is black.
  3) Photoreceptors: high-pass filter (adaptation, tau 250 ms).
  4) Motion: Hassenstein-Reichardt correlators between neighbouring ommatidia (low-pass delay 35 ms),
     the classic model of what T4/T5 compute. Four directions: front-to-back / back-to-front, up / down.
  5) Channels: forward (front-to-back, T4a/T5a), backward (back-to-front, T4b/T5b) and expansion
     (motion away from nadir, the looming signal pooled by LPLC2).
Only pitch is modelled; the eye is fixed to the body and faces downrange.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

SUN_ELEV = math.radians(10.8)  # sun elevation at the Apollo 11 landing
SUN_DIR = np.array([-math.cos(SUN_ELEV), math.sin(SUN_ELEV), 0.0])  # sun behind (east), LM flying west
SPACING = math.radians(5.0)


class LunarSurface:
    """Flat ground with crater shading and albedo. x downrange, z lateral (m)."""

    def __init__(self, seed=11, extent=4000.0):
        rng = np.random.default_rng(seed)
        # power law N(>r) ~ r^-2, 0.5-150 m
        r = 0.5 * (1 - rng.random(60000)) ** (-1 / 2.0)
        r = r[r < 150][:40000]
        n = len(r)
        cx = rng.uniform(-extent, extent * 0.3, n)
        cz = rng.uniform(-extent * 0.5, extent * 0.5, n)
        self.r = r
        self.c = np.stack([cx, cz], 1)
        self.tree = cKDTree(self.c)
        self.rmax = r.max()
        self.phase = rng.uniform(0, 2 * np.pi, (6, 2))

    def _albedo(self, x, z, footprint):
        a = np.full_like(x, 0.55)
        for i, wl in enumerate([200.0, 60.0, 18.0, 6.0, 2.0, 0.6]):
            k = 2 * np.pi / wl
            keep = wl >= 4 * footprint  # band limit: patterns narrower than ~2 ommatidia are blurred out
            a += keep * 0.06 * np.sin(k * x + self.phase[i, 0]) * np.sin(k * 0.83 * z + self.phase[i, 1])
        return a

    def luminance(self, x, z, footprint):
        """Luminance per point; footprint (m) is the ommatidium's field on the ground."""
        pts = np.stack([x, z], 1)
        gx = np.zeros_like(x)
        gz = np.zeros_like(x)
        footprint = np.broadcast_to(footprint, x.shape)
        lists = self.tree.query_ball_point(pts, r=np.full(len(x), self.rmax * 2.0))
        for i, js in enumerate(lists):
            if not js:
                continue
            js = np.asarray(js)
            rr = self.r[js]
            ok = rr > 1.0 * footprint[i]
            if not ok.any():
                continue
            js, rr = js[ok], rr[ok]
            d = pts[i] - self.c[js]
            dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
            u = dist / rr
            # bowl (depth 0.2r) + rim, slope dh/dd
            slope = np.where(u < 1, 0.4 * u, -0.25 * (u - 1) / 0.09 * np.exp(-((u - 1) ** 2) / 0.045))
            slope = np.where(u < 2, slope, 0.0)
            gx[i] = np.sum(slope * d[:, 0] / dist)
            gz[i] = np.sum(slope * d[:, 1] / dist)
        n = np.stack([-gx, np.ones_like(x), -gz], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        shade = np.clip(n @ SUN_DIR, 0.0, None) / math.sin(SUN_ELEV)
        return self._albedo(x, z, footprint) * (0.08 + 0.92 * np.clip(shade, 0, 3))


class FlyEye:
    def __init__(self, surface: LunarSurface):
        self.surface = surface
        az = np.arange(-180, 180, 5.0) + 2.5
        el = np.arange(-90, 0, 5.0) + 2.5  # lower half only; the upper half sees black sky
        self.az, self.el = np.meshgrid(np.radians(az), np.radians(el))  # (n_el, n_az)
        # body frame: X forward, Y up, Z right
        ce = np.cos(self.el)
        self.dirs = np.stack([ce * np.cos(self.az), np.sin(self.el), ce * np.sin(self.az)], -1)
        self.shape = self.az.shape
        self.reset()

    def reset(self):
        self.hp_state = None
        self.prev = None
        self.lp = None

    def render(self, x_downrange, h, pitch):
        """Ommatidium luminance (n_el, n_az). pitch: thrust-axis tilt downrange (rad).

        h: height above ground for LunarSurface; absolute height above the datum for Terrain
        (rays are cast against the 3D terrain and read from its shaded luminance map)."""
        c, s = math.cos(pitch), math.sin(pitch)
        d = self.dirs.reshape(-1, 3)
        # rotate about the lateral axis
        dx = c * d[:, 0] + s * d[:, 1]
        dy = -s * d[:, 0] + c * d[:, 1]
        dz = d[:, 2]
        img = np.full(len(d), 0.0)  # sky
        if hasattr(self.surface, "raycast"):
            t = self.surface.raycast(x_downrange, h, 0.0, dx, dy, dz)
            hit = np.isfinite(t)
            img[hit] = self.surface.sample_luminance(x_downrange + t[hit] * dx[hit], t[hit] * dz[hit], t[hit] * SPACING)
            # far ground (off-map / horizon) at mean luminance
            far = ~hit & (dy < -1e-3)
            img[far] = 1.0
            return img.reshape(self.shape)
        hit = dy < -1e-3
        hh = max(h, 0.3)
        t = hh / -dy[hit]
        footprint = t * SPACING
        img[hit] = self.surface.luminance(x_downrange + t * dx[hit], t * dz[hit], footprint)
        return img.reshape(self.shape)

    def motion(self, img, dt, tau_hp=0.25, tau_lp=0.035, tau_norm=0.5):
        """Hassenstein-Reichardt correlators with contrast normalisation.

        Returns (forward, backward, expansion, prog map, elevation map). forward/backward: positive/negative
        part of the mean front-to-back motion in the lateral-ventral field. expansion: mean motion away from
        nadir in the ventral field (front and back contributions of forward flight are meant to cancel).
        """
        if self.hp_state is None:
            self.hp_state = img.copy()
            self.lp = np.zeros_like(img)
            self.energy = np.full_like(img, 1e-4)
        a_hp = math.exp(-dt / tau_hp)
        self.hp_state = a_hp * self.hp_state + (1 - a_hp) * img
        p = img - self.hp_state  # photoreceptor high-pass (light adaptation)
        a_lp = math.exp(-dt / tau_lp)
        self.lp = a_lp * self.lp + (1 - a_lp) * p  # delayed arm
        a_n = math.exp(-dt / tau_norm)
        self.energy = a_n * self.energy + (1 - a_n) * p**2  # local contrast energy
        lp, cur, en = self.lp, p, self.energy
        # +azimuth and +elevation (towards horizon), normalised by neighbour energy
        norm_az = 0.5 * (en + np.roll(en, -1, axis=1)) + 1e-6
        m_az = (lp * np.roll(cur, -1, axis=1) - cur * np.roll(lp, -1, axis=1)) / norm_az
        m_el = np.zeros_like(cur)
        m_el[:-1] = (lp[:-1] * cur[1:] - cur[:-1] * lp[1:]) / (0.5 * (en[:-1] + en[1:]) + 1e-6)
        # front-to-back is +az on the right eye, -az on the left
        prog = np.where(self.az > 0, m_az, -m_az)
        side = (np.abs(self.az) > np.radians(20)) & (np.abs(self.az) < np.radians(160))
        ventral = self.el < np.radians(-15)
        f = float(np.mean(prog[side & ventral]))
        ex = float(np.mean(m_el[self.el < np.radians(-20)]))
        return max(f, 0.0), max(-f, 0.0), max(ex, 0.0), prog, m_el
