"""Interactive 2D Apollo 11 landing simulator (pygame).

    python -m apollo_sim.viewer          # from PDI
    python -m apollo_sim.viewer --p66    # from where Armstrong took over in P66

Keys:
  =        ROD up: descend 1 ft/s slower (first click enters P66, as Armstrong did)
  -        ROD down: descend 1 ft/s faster
  1..5     time warp 1x, 2x, 5x, 10x, 30x
  SPACE    pause      R  restart      ESC  quit
"""
import argparse
import math
import random

import numpy as np
import pygame

from . import constants as C
from . import reference as ref
from .agc import AGC
from .lm import at_ignition

W, H = 1280, 720
BG = (6, 8, 14)
GROUND = (92, 92, 98)
GROUND_DARK = (58, 58, 64)
FG = (225, 228, 235)
DIM = (130, 136, 150)
GREEN = (120, 230, 140)
AMBER = (255, 190, 80)
RED = (240, 90, 80)
DSKY_BG = (34, 36, 40)
WARPS = [1, 2, 5, 10, 30]
DT = 0.05
PROBE = 1.7
SHUTDOWN_DELAY = 1.0


def get_str(t_from_pdi):
    s = ref.PDI + t_from_pdi
    return f"{int(s // 3600):03d}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"


class Sim:
    def __init__(self, p66: bool, terrain=None):
        self.lm = at_ignition(terrain)
        self.agc = AGC(self.lm)
        self.t = 0.0
        self.contact_t = None
        self.events = []
        if p66:
            while self.lm.altitude > 470 * C.FT:
                self.step()
            self.agc.start_p66()
            if terrain is not None:  # with terrain, horizontal speed is yours too (ATT HOLD)
                self.agc.horizontal_nulling = False
            self.events.append((self.t, "P66 - you have control"))

    def step(self):
        lm = self.lm
        if lm.s.landed:
            return
        prev = self.agc.program
        self.agc.update(self.t)
        if self.agc.program != prev:
            self.events.append((self.t, f"P{self.agc.program}"))
        if self.contact_t is None and lm.altitude <= PROBE:
            self.contact_t = self.t
            self.events.append((self.t, "CONTACT LIGHT"))
        if self.contact_t is not None and self.t - self.contact_t >= SHUTDOWN_DELAY and lm.engine_on:
            lm.engine_on = False
            self.events.append((self.t, "SHUTDOWN"))
        lm.step(DT)
        self.t += DT
        if lm.s.landed:
            vd, vh = lm.s.touchdown_v
            slope = lm.touchdown_slope if lm.terrain is not None else 0.0
            ok = vd <= 10 * C.FT and abs(vh) <= 4 * C.FT and slope <= 12.0
            self.events.append((self.t, f"{'EAGLE HAS LANDED' if ok else 'CRASH'}  {vd / C.FT:.1f} ft/s, slope {slope:.0f}°"))


class Terrain:
    """Decorative seeded surface profile for the 2D view (the physics uses flat ground here)."""

    def __init__(self, seed=11):
        rng = random.Random(seed)
        self.craters = [(rng.uniform(-3000, 3000), rng.uniform(4, 60)) for _ in range(160)]
        self.rocks = [(rng.uniform(-400, 400), rng.uniform(0.3, 2.0)) for _ in range(120)]

    def height(self, x):
        y = 0.0
        for cx, r in self.craters:
            d = (x - cx) / r
            if abs(d) < 1.3:
                y += r * 0.18 * (math.cos(min(abs(d), 1.0) * math.pi) * 0.5 - 0.5) if abs(d) < 1 else r * 0.03
        return y


def lm_polygons(scale):
    """LM side view in metres, body axis up. Landing gear ~9.4 m wide."""
    s = scale
    descent = [(-2.1, 1.0), (2.1, 1.0), (2.1, 2.7), (-2.1, 2.7)]
    ascent = [(-1.9, 2.7), (1.9, 2.7), (1.6, 5.4), (0.3, 6.2), (-1.5, 5.6), (-1.9, 4.0)]
    legs = [[(-2.1, 1.6), (-4.2, 0.0)], [(2.1, 1.6), (4.2, 0.0)], [(-2.1, 2.3), (-4.2, 0.0)], [(2.1, 2.3), (4.2, 0.0)]]
    pads = [[(-4.7, 0.0), (-3.7, 0.0)], [(3.7, 0.0), (4.7, 0.0)]]
    probes = [[(-4.2, 0.0), (-4.2, -1.7)], [(4.2, 0.0), (4.2, -1.7)]]
    bell = [(-0.9, 1.0), (0.9, 1.0), (0.6, 0.2), (-0.6, 0.2)]
    sc = lambda pts: [(x * s, y * s) for x, y in pts]
    return sc(descent), sc(ascent), [sc(l) for l in legs], [sc(p) for p in pads], [sc(p) for p in probes], sc(bell)


class Viewer:
    def __init__(self, p66):
        pygame.init()
        pygame.display.set_caption("Apollo 11 - Eagle landing")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("consolas", 30, bold=True)
        self.seg = pygame.font.SysFont("consolas", 34, bold=True)
        self.p66_start = p66
        self.terrain = Terrain()
        self.stars = [(random.randrange(W), random.randrange(H), random.random()) for _ in range(250)]
        self.reset()

    def reset(self):
        self.sim = Sim(self.p66_start)
        self.warp_i = 0 if self.p66_start else 3
        self.paused = False
        self.downrange0 = None

    def draw_scene(self):
        lm = self.lm
        alt = max(lm.altitude, 0.0)
        mpp = max(alt / 200.0, 0.045)  # m per pixel; zooms in near the ground
        cx, cy = W * 0.38, H * 0.45
        ground_y = cy + alt / mpp
        # stars
        for x, y, b in self.stars:
            if y < ground_y:
                c = int(60 + 140 * b)
                self.screen.set_at((x, y), (c, c, c))
        # surface
        dr = lm.s.pos[0]
        if ground_y < H + 50:
            pygame.draw.rect(self.screen, GROUND, (0, ground_y, W, H - ground_y + 10))
            for kx, kr in self.terrain.craters:
                px = cx + (kx - dr) / mpp
                w = 2 * kr / mpp
                if -w < px < W + w and w > 3:
                    hgt = min(0.35 * kr / mpp, 28)
                    pygame.draw.ellipse(self.screen, GROUND_DARK, (px - w / 2, ground_y + 2, w, max(2, hgt)))
            for rx, rs in self.terrain.rocks:
                px = cx + (rx - dr) / mpp
                if -20 < px < W + 20 and rs / mpp > 1:
                    pygame.draw.circle(self.screen, GROUND_DARK, (int(px), int(ground_y + rs / mpp * 0.4)), int(rs / mpp))
            # landing site marker
            sx = cx + (0 - dr) / mpp
            if -20 < sx < W + 20:
                pygame.draw.line(self.screen, AMBER, (sx, ground_y - 6), (sx, ground_y + 6), 2)
        # LM
        up_local = lm.s.pos / np.linalg.norm(lm.s.pos)
        local_tilt = math.atan2(up_local[0], up_local[1])  # local vertical vs. landing-site vertical
        tilt = lm.s.pitch - local_tilt
        scale = max(1.0 / mpp, 2.5)  # exaggerate the LM when high so it stays visible
        descent, ascent, legs, pads, probes, bell = lm_polygons(scale)
        base_y = cy  # footpads
        rot = lambda p: (cx + p[0] * math.cos(tilt) + p[1] * math.sin(tilt),
                         base_y - (-p[0] * math.sin(tilt) + p[1] * math.cos(tilt)))
        if lm.s.thrust > 100 and lm.engine_on:
            frac = lm.s.thrust / C.DPS_NOMINAL_MAX
            ln = (4 + 14 * frac) * scale * (0.9 + 0.2 * random.random())
            flame = [rot((-0.6 * scale, 0.2 * scale)), rot((0.6 * scale, 0.2 * scale)), rot((0, 0.2 * scale - ln))]
            pygame.draw.polygon(self.screen, (255, 170, 60), flame)
            inner = [rot((-0.3 * scale, 0.2 * scale)), rot((0.3 * scale, 0.2 * scale)), rot((0, 0.2 * scale - ln * 0.6))]
            pygame.draw.polygon(self.screen, (255, 240, 200), inner)
            if alt < 30:  # dust
                for _ in range(int(40 * frac * (30 - alt) / 30)):
                    x = cx + random.gauss(0, 12 * scale)
                    pygame.draw.circle(self.screen, (150, 150, 150), (int(x), int(ground_y - random.random() * 3)), 1)
        pygame.draw.polygon(self.screen, (200, 170, 70), [rot(p) for p in descent])
        pygame.draw.polygon(self.screen, (200, 200, 205), [rot(p) for p in ascent])
        pygame.draw.polygon(self.screen, (90, 90, 95), [rot(p) for p in bell])
        for seg in legs + pads:
            pygame.draw.line(self.screen, (220, 220, 220), rot(seg[0]), rot(seg[1]), max(1, int(0.25 * scale)))
        probe_len = min(1.7, alt)  # probes bend on contact
        for seg in probes:
            tip = (seg[1][0], -probe_len * scale)
            pygame.draw.line(self.screen, (180, 180, 180), rot(seg[0]), rot(tip), 1)
        # scale bar
        bar = 10 ** math.floor(math.log10(max(mpp * 150, 1e-3)))
        pygame.draw.line(self.screen, DIM, (20, H - 24), (20 + bar / mpp, H - 24), 2)
        self.text(f"{bar:g} m", 20, H - 44, DIM, self.small)

    def text(self, s, x, y, col=FG, font=None):
        self.screen.blit((font or self.font).render(s, True, col), (x, y))

    def draw_dsky(self):
        lm, agc = self.lm, self.sim.agc
        x0, y0 = W - 330, 20
        pygame.draw.rect(self.screen, DSKY_BG, (x0, y0, 310, 250), border_radius=8)
        self.text("PROG", x0 + 205, y0 + 10, DIM, self.small)
        self.text(f"{agc.program:02d}", x0 + 205, y0 + 26, GREEN, self.seg)
        verb, noun = (6, 63) if agc.program in (63, 64) else (6, 60)
        if agc.program == 64:
            noun = 64
        self.text("VERB", x0 + 20, y0 + 70, DIM, self.small)
        self.text("NOUN", x0 + 205, y0 + 70, DIM, self.small)
        self.text(f"{verb:02d}", x0 + 20, y0 + 86, GREEN, self.seg)
        self.text(f"{noun:02d}", x0 + 205, y0 + 86, GREEN, self.seg)
        speed = math.hypot(lm.vertical_speed(), lm.horizontal_speed()) / C.FT
        hdot = lm.vertical_speed() / C.FT
        h = max(lm.altitude, 0) / C.FT
        if noun == 63:
            r1 = f"{speed:+06.0f}"
        elif noun == 64:  # real R1 shows TTF and LPD angle; LPD is not modelled
            r1 = f"+{max(0, -agc.ttf):02.0f}   "
        else:
            r1 = f"{lm.horizontal_speed() / C.FT:+06.1f}"
        for i, r in enumerate([r1, f"{hdot:+06.1f}", f"{h:+06.0f}"]):
            pygame.draw.line(self.screen, (60, 64, 70), (x0 + 15, y0 + 130 + 38 * i), (x0 + 295, y0 + 130 + 38 * i))
            self.text(r, x0 + 120, y0 + 133 + 38 * i, GREEN, self.seg)
        self.text("R1", x0 + 20, y0 + 145, DIM, self.small)
        self.text("R2 HDOT ft/s", x0 + 20, y0 + 183, DIM, self.small)
        self.text("R3 H ft", x0 + 20, y0 + 221, DIM, self.small)

    def draw_hud(self):
        lm, agc, sim = self.lm, self.sim.agc, self.sim
        s = lm.s
        frac = s.thrust / C.DPS_NOMINAL_MAX * 100
        fuel_pct = s.propellant / C.PROPELLANT_PDI * 100
        lines = [
            (f"GET {get_str(sim.t)}   PDI+{sim.t:6.1f} s", FG),
            (f"altitude   {max(lm.altitude, 0):9.1f} m   ({max(lm.altitude, 0) / C.FT:8.0f} ft)", FG),
            (f"vertical   {lm.vertical_speed():9.2f} m/s ({lm.vertical_speed() / C.FT:+7.1f} ft/s)", FG),
            (f"horizontal {lm.horizontal_speed():9.2f} m/s ({lm.horizontal_speed() / C.FT:+7.1f} ft/s)", FG),
            (f"thrust     {frac:6.1f} %  {'FTP' if s.thrust > 0.9 * C.DPS_FTP else ''}", FG),
            (f"fuel       {s.propellant:7.0f} kg  ({fuel_pct:4.1f} %)", RED if fuel_pct < C.LOW_LEVEL_FRACTION * 100 else FG),
            (f"mass       {s.mass:7.0f} kg", DIM),
        ]
        if agc.program == 66:
            lines.append((f"ROD cmd    {agc.v_desired / C.FT:+6.1f} ft/s", AMBER))
        for i, (ln, col) in enumerate(lines):
            self.text(ln, 20, 20 + 24 * i, col)
        warp = WARPS[self.warp_i]
        self.text(f"time x{warp}{'  PAUSED' if self.paused else ''}", 20, 20 + 24 * len(lines) + 8, DIM)
        self.text("= / - : ROD   1-5: warp   SPACE: pause   R: restart", W - 420, H - 28, DIM, self.small)
        for i, (t, ev) in enumerate(self.sim.events[-6:]):
            self.text(f"{get_str(t)}  {ev}", W - 330, 290 + 22 * i, AMBER, self.small)
        if s.landed:
            vd, vh = s.touchdown_v
            ok = vd <= 10 * C.FT and abs(vh) <= 4 * C.FT
            msg = "THE EAGLE HAS LANDED" if ok else "CRASH"
            self.text(msg, W * 0.38 - 190, H - 170, GREEN if ok else RED, self.big)
            self.text(f"touchdown {vd / C.FT:.1f} ft/s vertical, {abs(vh) / C.FT:.1f} ft/s lateral | fuel {s.propellant:.0f} kg",
                      W * 0.38 - 190, H - 125, FG)
            self.text(f"Armstrong (Apollo 11): 1.7 ft/s vertical, ~{ref.PROPELLANT_LEFT:.0f} kg fuel",
                      W * 0.38 - 190, H - 100, DIM)

    @property
    def lm(self):
        return self.sim.lm

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_r:
                        self.reset()
                    elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        if self.sim.agc.program != 66:
                            self.sim.events.append((self.sim.t, "P66 - you have control"))
                        self.sim.agc.rod_click(+1)
                    elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        if self.sim.agc.program != 66:
                            self.sim.events.append((self.sim.t, "P66 - you have control"))
                        self.sim.agc.rod_click(-1)
                    elif pygame.K_1 <= e.key <= pygame.K_5:
                        self.warp_i = e.key - pygame.K_1
            dt_real = self.clock.tick(60) / 1000.0
            if not self.paused and not self.lm.s.landed:
                for _ in range(max(1, round(dt_real * WARPS[self.warp_i] / DT))):
                    self.sim.step()
            self.screen.fill(BG)
            self.draw_scene()
            self.draw_dsky()
            self.draw_hud()
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p66", action="store_true", help="start where Armstrong entered P66")
    Viewer(ap.parse_args().p66).run()
