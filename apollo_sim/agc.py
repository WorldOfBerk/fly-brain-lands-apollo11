"""Lunar landing guidance: P63/P64/P65/P66 logic of Luminary099 in Python.

Source: Luminary099/LUNAR_LANDING_GUIDANCE_EQUATIONS.agc
  - TTF/8CL (line 499): TTF from the cubic in the downrange component
        J*T^3 + 6*A*T^2 + (6*VG + 18*VD)*T + 24*(RD - RG) = 0      (T < 0)
  - QUADGUID (line 540): ACG = ADG + 6*(VDG + VG)/TTF + 12*(RDG - RG)/TTF^2
  - AFCCALC (line 610): AFC = ACG - g; if |AFC|*m > HIGHESTF the horizontal part is cut
  - EXTLOGIC (line 702): TTF > -TENDBRAK -> P64, TTF > -TENDAPPR -> P65
  - P65VERT (line 917): ACG = (V2FG - VG) / TAUVERT
  - P66 (RODCOMP, line 958): each ROD click changes the desired sink rate by RODSCALE,
        thrust closes the rate error with time constant TAUROD
Throttle logic (THROTTLE_CONTROL_ROUTINES.agc): above HIGHCRIT -> FTP, below LOWCRIT -> throttleable.

Simplifications: perfect navigation (no 2900 ft altitude error as on Apollo 11), fixed guidance frame,
no LEADTIME compensation.
"""
import numpy as np

from . import constants as C


class AGC:
    def __init__(self, lm):
        self.lm = lm
        self.program = 63
        self.ttf = C.TTF_IGN  # initial guess
        self.target = C.BRAKING
        self.throttle_ftp = True  # P63 starts at FTP
        self.next_pass = C.ZOOMTIME
        self.v_desired = 0.0  # P66 desired vertical speed
        self.rod_count = 0
        self.horizontal_nulling = True  # P66: auto-null horizontal speed; False: pilot sets tilt
        self.tilt_cmd = 0.0  # rad, tilt held in P66 ATT HOLD
        self.log = []

    def guidance_state(self):
        s = self.lm.s
        rg = np.array([s.pos[1] - C.R_MOON, s.pos[0]])  # (x up, z downrange)
        # landing radar updates altitude relative to the terrain below (no nav error modelled)
        rg[0] -= self.lm.ground_height
        vg = np.array([s.vel[1], s.vel[0]])
        r = np.linalg.norm(s.pos)
        g = -C.MU_MOON * s.pos / r**3
        gg = np.array([g[1], g[0]])
        return rg, vg, gg

    def _solve_ttf(self, rg, vg):
        t = self.target
        J, A, V, R = t["J"], t["A"][1], t["V"][1], t["R"][1]
        f = lambda T: J * T**3 + 6 * A * T**2 + (6 * vg[1] + 18 * V) * T + 24 * (R - rg[1])
        df = lambda T: 3 * J * T**2 + 12 * A * T + (6 * vg[1] + 18 * V)
        T = min(self.ttf, -1.0)
        for _ in range(30):  # Newton (ROOTPSRS in the AGC)
            d = df(T)
            if abs(d) < 1e-9:
                break
            T_new = T - f(T) / d
            if T_new >= 0:
                T_new = T / 2
            if abs(T_new - T) < 1e-4:
                T = T_new
                break
            T = T_new
        return T

    def _quadguid(self, rg, vg):
        t = self.target
        T = self.ttf
        R, V, A = np.array(t["R"]), np.array(t["V"]), np.array(t["A"])
        return A + 6 * (V + vg) / T + 12 * (R - rg) / T**2

    def _command(self, acg, gg):
        """AFC = ACG - g, thrust limit, attitude and throttle commands."""
        m = self.lm.s.mass
        afc = acg - gg
        fmax = C.HIGHESTF
        if np.linalg.norm(afc) * m > fmax:  # AFCCALC2: keep vertical, cut horizontal
            ax = afc[0]
            hmax2 = (fmax / m) ** 2 - ax**2
            afc[1] = np.sign(afc[1]) * np.sqrt(max(hmax2, 0.0))
        force = np.linalg.norm(afc) * m
        self.lm.pitch_cmd = float(np.arctan2(afc[1], afc[0]))
        self._throttle(force)
        return afc

    def _throttle(self, force):
        frac = force / C.DPS_NOMINAL_MAX
        if frac > C.HIGHCRIT:
            self.throttle_ftp = True
        elif frac < C.LOWCRIT:
            self.throttle_ftp = False
        if self.throttle_ftp:
            self.lm.thrust_cmd = C.DPS_FTP
        else:
            self.lm.thrust_cmd = float(np.clip(frac, C.DPS_THROTTLE_MIN, C.DPS_THROTTLE_MAX) * C.DPS_NOMINAL_MAX)

    def update(self, t):
        s = self.lm.s
        if s.landed:
            return
        if t < C.ZOOMTIME:  # ignition: 10 % thrust, retrograde
            self.lm.thrust_cmd = C.DPS_THROTTLE_MIN * C.DPS_NOMINAL_MAX
            self.lm.pitch_cmd = -np.pi / 2
            return
        if self.program in (66, 65):
            self._vertical_mode(t)
            return
        if t < self.next_pass:
            return
        self.next_pass = t + C.GUIDANCE_PERIOD
        rg, vg, gg = self.guidance_state()
        self.ttf = self._solve_ttf(rg, vg)
        if self.program == 63 and self.ttf > -C.TENDBRAK:
            self.program, self.target, self.ttf = 64, C.APPROACH, C.TTF_P64
            self.ttf = self._solve_ttf(rg, vg)
        elif self.program == 64 and self.ttf > -C.TENDAPPR:
            self.program = 65
            self._vertical_mode(t)
            return
        acg = self._quadguid(rg, vg)
        self._command(acg, gg)
        self.log.append((t, self.program, self.ttf))

    def start_p66(self):
        """First ROD click: enter P66 with desired rate = current vertical speed (STARTP66)."""
        if self.program != 66:
            self.program = 66
            self.v_desired = self.lm.vertical_speed()
            self.tilt_cmd = self.lm.s.pitch  # ATT HOLD

    def set_tilt(self, angle: float, limit=np.radians(15.0)):
        """Proportional ACA: desired tilt (rad), tracked by the rate-limited attitude loop."""
        self.tilt_cmd = float(np.clip(angle, -limit, limit))

    def aca_click(self, n: int, step=np.radians(1.0), limit=np.radians(15.0)):
        """ACA in ATT HOLD: +1 tilts thrust downrange (speed up), -1 back (brake)."""
        self.tilt_cmd = float(np.clip(self.tilt_cmd + n * step, -limit, limit))

    def rod_click(self, n: int):
        """+1: slower descent, -1: faster (channel 16 bits 6/7)."""
        if n == 0:
            return
        self.start_p66()
        self.v_desired = min(self.v_desired + n * C.RODSCALE, 0.0)

    def _vertical_mode(self, t):
        rg, vg, gg = self.guidance_state()
        if self.program == 65:  # auto descent at a small sink rate
            target_v = np.array([-1.0, 0.0])
            acg = (target_v - vg) / 3.0
        else:
            acg = np.zeros(2)
            acg[0] = (self.v_desired - vg[0]) / C.TAUROD
            if self.horizontal_nulling:
                acg[1] = -vg[1] / 6.0
            else:  # thrust axis at the pilot's tilt, vertical part from the ROD loop
                afc_x = acg[0] - gg[0]
                acg[1] = afc_x * np.tan(self.tilt_cmd) + gg[1]
                self._command(acg, gg)
                return
        acg[1] = np.clip(acg[1], -1.0, 1.0)  # horizontal accel limit (~ +-20 deg tilt)
        self._command(acg, gg)
