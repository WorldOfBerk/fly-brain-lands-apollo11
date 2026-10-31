"""Lunar terrain around Tranquility Base: height map, sun shadows and luminance.

Physics (altitude, touchdown, slope), the fly's eye (ray casting) and the 3D view all use this map.

Coordinates: x downrange (the LM flies west, towards +x), z lateral, y up. (0, 0) is where Eagle landed.
Height is 0 outside the map.

Contents:
  - West crater: 190 m diameter, centre 550 m east of the landing site; Armstrong flew over it at ~100 m;
    boulders concentrated east-northeast in its ejecta (LROC; Chandrayaan-2 boulder study, 2024).
  - Little West: ~30 m diameter, ~60 m east of the site.
  - Small craters: equilibrium density for mare N(>D) ~ 0.08 * D^-2 km^-2 (D in km), 2-120 m,
    with random degradation (young craters deep, old ones shallow).
  - Fresh simple crater shape (Pike 1977): depth/diameter ~ 0.2, rim height ~ 0.04 D, ejecta ~ (r/R)^-3.
  - Low-frequency undulation of a few metres.
  - Sun elevation 10.8 deg (Apollo 11 landing), sun in the east (behind the LM, -x).
"""
import math
from dataclasses import dataclass

import numpy as np

SUN_ELEV = math.radians(10.8)
SUN_DIR = np.array([-math.cos(SUN_ELEV), math.sin(SUN_ELEV), 0.0])  # ground -> sun


@dataclass
class TerrainSpec:
    x_min: float = -2600.0
    x_max: float = 900.0
    z_min: float = -700.0
    z_max: float = 700.0
    res: float = 0.5  # m per pixel
    seed: int = 1969
    d_min: float = 2.0  # m, smallest crater stamped into the map
    d_max: float = 120.0


class Terrain:
    def __init__(self, spec: TerrainSpec | None = None):
        self.s = spec or TerrainSpec()
        s = self.s
        self.nx = int((s.x_max - s.x_min) / s.res) + 1
        self.nz = int((s.z_max - s.z_min) / s.res) + 1
        self.rng = np.random.default_rng(s.seed)
        self.h = np.zeros((self.nz, self.nx), np.float32)  # [z, x]
        self.craters = []  # (x, z, D)
        self.boulders = []  # (x, z, diameter)
        self._undulation()
        self._random_craters()
        self._named_craters()
        self._boulders()
        self._albedo_and_light()

    def _grid(self, x0, x1, z0, z1):
        s = self.s
        i0 = max(0, int((x0 - s.x_min) / s.res))
        i1 = min(self.nx, int((x1 - s.x_min) / s.res) + 2)
        j0 = max(0, int((z0 - s.z_min) / s.res))
        j1 = min(self.nz, int((z1 - s.z_min) / s.res) + 2)
        xs = s.x_min + np.arange(i0, i1) * s.res
        zs = s.z_min + np.arange(j0, j1) * s.res
        return slice(j0, j1), slice(i0, i1), xs, zs

    def _undulation(self):
        s = self.s
        xs = s.x_min + np.arange(self.nx) * s.res
        zs = s.z_min + np.arange(self.nz) * s.res
        X, Z = np.meshgrid(xs, zs)
        for wl, amp in [(1500, 3.0), (600, 1.5), (250, 0.7), (100, 0.3)]:
            k = 2 * np.pi / wl
            ph = self.rng.uniform(0, 2 * np.pi, 4)
            self.h += amp * (np.sin(k * X + ph[0]) * np.cos(0.7 * k * Z + ph[1])
                             + 0.5 * np.sin(0.6 * k * X + 1.3 * k * Z + ph[2])).astype(np.float32)

    def stamp_crater(self, cx, cz, D, freshness=1.0):
        """Pike (1977) simple crater, shallower when degraded. freshness 1 = fresh, 0.2 = heavily degraded."""
        R = D / 2
        depth = 0.2 * D * freshness
        rim = 0.04 * D * freshness
        sl_z, sl_x, xs, zs = self._grid(cx - 3 * R, cx + 3 * R, cz - 3 * R, cz + 3 * R)
        if len(xs) == 0 or len(zs) == 0:
            return
        X, Z = np.meshgrid(xs, zs)
        u = np.hypot(X - cx, Z - cz) / R
        inside = -depth + (depth + rim) * u**2
        outside = rim * np.clip(u, 1, None) ** -3 * np.clip((3 - u) / 2, 0, 1)
        self.h[sl_z, sl_x] += np.where(u < 1, inside, outside).astype(np.float32)
        self.craters.append((cx, cz, D))

    def _random_craters(self):
        s = self.s
        area_km2 = (s.x_max - s.x_min) * (s.z_max - s.z_min) / 1e6
        n = int(0.08 * (s.d_min / 1000) ** -2 * area_km2 - 0.08 * (s.d_max / 1000) ** -2 * area_km2)
        u = self.rng.random(n)
        # N(>D) ~ D^-2, inverse transform sampling
        D = 1 / np.sqrt(u / s.d_min**2 + (1 - u) / s.d_max**2)
        order = np.argsort(-D)  # large first, small ones on top
        xs = self.rng.uniform(s.x_min, s.x_max, n)
        zs = self.rng.uniform(s.z_min, s.z_max, n)
        fresh = np.clip(self.rng.beta(1.2, 2.5, n) + 0.15, 0.15, 1.0)
        for k in order:
            self.stamp_crater(xs[k], zs[k], D[k], fresh[k])

    def _named_craters(self):
        self.stamp_crater(-550.0, 20.0, 190.0, freshness=1.0)  # West
        self.stamp_crater(-60.0, 8.0, 30.0, freshness=0.9)  # Little West
        self.named = {"West": (-550.0, 20.0, 190.0), "Little West": (-60.0, 8.0, 30.0)}

    def _boulders(self):
        """Boulders in West's ejecta, mostly east (-x) and north. Half-buried domes."""
        cx, cz, D = self.named["West"]
        R = D / 2
        n = 900
        ang = self.rng.vonmises(math.radians(160), 1.2, n)  # from +x; 180 = east
        rad = R * (1.0 + self.rng.gamma(1.6, 0.35, n))
        size = np.clip(0.6 * (1 - self.rng.random(n)) ** (-1 / 2.2), 0.6, 6.0)
        for a, r, d in zip(ang, rad, size):
            bx, bz = cx + r * math.cos(a), cz + r * math.sin(a)
            rr = d / 2
            sl_z, sl_x, xs, zs = self._grid(bx - rr, bx + rr, bz - rr, bz + rr)
            X, Z = np.meshgrid(xs, zs)
            q = 1 - ((X - bx) ** 2 + (Z - bz) ** 2) / rr**2
            self.h[sl_z, sl_x] += (0.7 * rr * np.sqrt(np.clip(q, 0, None))).astype(np.float32)
            self.boulders.append((bx, bz, d))

    def _albedo_and_light(self):
        s = self.s
        gz, gx = np.gradient(self.h, s.res)
        n = np.stack([-gx, np.ones_like(gx), -gz], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        self.normal = n.astype(np.float32)
        lambert = np.clip(n @ SUN_DIR.astype(np.float32), 0, None) / math.sin(SUN_ELEV)
        # shadows: sun towards -x, sweep a horizon line along +x
        shadow = np.zeros_like(self.h, dtype=bool)
        depth = np.zeros_like(self.h)  # how far above the ground the sun ray passes (m)
        horizon = np.full(self.nz, -1e9, np.float32)
        drop = s.res * math.tan(SUN_ELEV)
        for i in range(self.nx):
            col = self.h[:, i]
            horizon = np.maximum(horizon - drop, col)
            depth[:, i] = horizon - col
            shadow[:, i] = col < horizon - 1e-3
        self.shadow = shadow
        self.shadow_depth = depth.astype(np.float32)
        # albedo: mare regolith ~0.07-0.10, fresh ejecta slightly brighter
        xs = s.x_min + np.arange(self.nx) * s.res
        zs = s.z_min + np.arange(self.nz) * s.res
        X, Z = np.meshgrid(xs, zs)
        alb = 0.085 + 0.006 * np.sin(X / 37.0 + 1.0) * np.sin(Z / 53.0)
        cx, cz, D = self.named["West"]
        alb += 0.025 * np.exp(-((np.hypot(X - cx, Z - cz) / (1.4 * D)) ** 2))
        self.albedo = alb.astype(np.float32)
        self.luminance = (self.albedo * lambert * ~shadow / 0.085).astype(np.float32)
        self._mips = [self.luminance]
        lv = self.luminance
        while min(lv.shape) > 8:
            lv = 0.25 * (lv[:-1:2, :-1:2] + lv[1::2, :-1:2] + lv[:-1:2, 1::2] + lv[1::2, 1::2])
            self._mips.append(lv)

    def _bilinear(self, arr, x, z, res):
        s = self.s
        fx = (np.asarray(x) - s.x_min) / res
        fz = (np.asarray(z) - s.z_min) / res
        ny, nx = arr.shape
        inside = (fx >= 0) & (fx < nx - 1) & (fz >= 0) & (fz < ny - 1)
        fx = np.clip(fx, 0, nx - 1.001)
        fz = np.clip(fz, 0, ny - 1.001)
        i, j = fx.astype(int), fz.astype(int)
        tx, tz = fx - i, fz - j
        v = (arr[j, i] * (1 - tx) * (1 - tz) + arr[j, i + 1] * tx * (1 - tz)
             + arr[j + 1, i] * (1 - tx) * tz + arr[j + 1, i + 1] * tx * tz)
        return np.where(inside, v, 0.0), inside

    def height(self, x, z=0.0):
        v, _ = self._bilinear(self.h, x, z, self.s.res)
        return v

    def slope_under(self, x, z=0.0, span=9.4):
        """Max slope (deg) under the landing gear (~9.4 m span)."""
        r = span / 2
        hx = (self.height(x + r, z) - self.height(x - r, z)) / span
        hz = (self.height(x, z + r) - self.height(x, z - r)) / span
        return float(np.degrees(np.arctan(np.hypot(hx, hz))))

    def sample_luminance(self, x, z, footprint):
        """Band-limited (mipmapped) luminance for a given footprint. Off-map: mean flat ground."""
        lvl = np.clip(np.log2(np.maximum(footprint, 1e-6) / self.s.res), 0, len(self._mips) - 1)
        out = np.zeros_like(lvl, dtype=np.float32)
        lo = np.floor(lvl).astype(int)
        for L in np.unique(lo):
            m = lo == L
            a, ins = self._bilinear(self._mips[L], x[m], z[m], self.s.res * 2**L)
            b, _ = self._bilinear(self._mips[min(L + 1, len(self._mips) - 1)], x[m], z[m], self.s.res * 2 ** min(L + 1, len(self._mips) - 1))
            w = lvl[m] - L
            out[m] = np.where(ins, (1 - w) * a + w * b, 1.0)
        return out

    def raycast(self, ox, oy, oz, dx, dy, dz, t_max=4000.0, steps=96):
        """First terrain hit distance t per ray (inf if none). Vectorised, adaptive step."""
        t = np.zeros_like(dx)
        hit = np.zeros(dx.shape, bool)
        alive = dy < 0  # upward rays never hit nearby ground
        for _ in range(steps):
            px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
            gap = py - self.height(px, pz)
            newly = alive & (gap <= 0.02)
            hit |= newly
            alive &= ~newly & (t < t_max)
            if not alive.any():
                break
            # safe step assuming slopes < ~45 deg
            t = np.where(alive, t + np.maximum(gap * 0.6, 0.05 + 0.002 * t), t)
        return np.where(hit, t, np.inf)
