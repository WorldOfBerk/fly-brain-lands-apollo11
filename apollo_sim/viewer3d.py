"""Interactive 3D landing simulator: terrain (West crater, boulders, shadows) and an OpenGL LM.

    python -m apollo_sim.viewer3d --p66     # take over where Armstrong did, both axes are yours
    python -m apollo_sim.viewer3d           # automatic landing from PDI

Keys:
  = / -          ROD: descend 1 ft/s slower / faster
  left / right   ACA (ATT HOLD): tilt thrust 1 deg back (brake) / forward
  C              camera: chase <-> Aldrin's window (like the 16 mm camera)
  E              toggle the fly's-eye panel
  1..5 warp, SPACE pause, R restart, ESC quit
"""
import argparse

import numpy as np
import pygame

from . import constants as C
from .eyeview import draw_eye
from .flyeye import FlyEye
from .render3d import Renderer
from .terrain import Terrain
from .viewer import DT, FG, DIM, W, H, Sim, Viewer


class Viewer3D(Viewer):
    def __init__(self, p66):
        print("building terrain (~10 s)...")
        self.terrain = Terrain()
        self.renderer = Renderer(self.terrain, W, H)
        self.camera = "chase"
        self.show_eye = True
        self.eye = FlyEye(self.terrain)
        self.eye_state = None
        super().__init__(p66)
        pygame.display.set_caption("Apollo 11 - Eagle landing (3D)")

    def reset(self):
        self.sim = Sim(self.p66_start, self.terrain)
        self.warp_i = 0 if self.p66_start else 3
        self.paused = False
        self.eye.reset()

    def draw_scene(self):
        lm = self.lm
        frac = lm.s.thrust / C.DPS_NOMINAL_MAX if lm.engine_on else 0.0
        img = self.renderer.render(lm.s.pos[0], lm.datum_altitude, lm.s.pitch, frac, self.camera,
                                   dt=0.0 if self.paused else DT * WARPS_ACTIVE(self))
        self.screen.blit(pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), (0, 0))
        panel = pygame.Surface((470, 250), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (10, 10))
        if self.show_eye:
            img_eye = self.eye.render(lm.s.pos[0], lm.datum_altitude, 0.0)
            _, _, _, prog, m_el = self.eye.motion(img_eye, DT)
            box = pygame.Surface((560, 170), pygame.SRCALPHA)
            box.fill((0, 0, 0, 170))
            self.screen.blit(box, (10, H - 205))
            self.text("fly's eye (1296 ommatidia, lower half)", 20, H - 200, DIM, self.small)
            draw_eye(self.screen, (20, H - 182, 540, 130), img_eye, prog, m_el, arrows_every=4)

    def draw_hud(self):
        super().draw_hud()
        agc = self.sim.agc
        if agc.program == 66 and not agc.horizontal_nulling:
            self.text(f"ACA tilt   {np.degrees(agc.tilt_cmd):+5.1f} deg", 20, 20 + 24 * 8 + 30, FG)
        self.text(f"camera: {'chase' if self.camera == 'chase' else 'Aldrin window'}   C: switch  E: fly eye  <-/->: ACA",
                  W - 640, H - 48, DIM, self.small)

    def run(self):
        orig = pygame.event.get

        def events():
            evs = orig()
            for e in evs:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_c:
                        self.camera = "window" if self.camera == "chase" else "chase"
                    elif e.key == pygame.K_e:
                        self.show_eye = not self.show_eye
                    elif e.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        agc = self.sim.agc
                        if agc.program != 66:
                            agc.start_p66()
                            agc.horizontal_nulling = False
                            self.sim.events.append((self.sim.t, "P66 - you have control"))
                        agc.aca_click(-1 if e.key == pygame.K_LEFT else 1)
            return evs

        pygame.event.get = events
        try:
            super().run()
        finally:
            pygame.event.get = orig


def WARPS_ACTIVE(v):
    from .viewer import WARPS
    return WARPS[v.warp_i]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p66", action="store_true", help="start where Armstrong entered P66")
    Viewer3D(ap.parse_args().p66).run()
