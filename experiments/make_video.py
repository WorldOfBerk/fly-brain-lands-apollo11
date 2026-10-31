"""Renders the video (1920x1080, 30 fps).

  1) title
  2) PDI -> P66 flown by Luminary guidance, 60x (2D profile) / 10x (3D, below 1500 m)
  3) P66 flown by the fly (replay of results/phase3c trial 0), 1.5x
  4) touchdown and comparison with Armstrong

Layout: 3D view (1280x720) + DSKY top left, brain activity right (640x1080, real soma positions),
fly's eye bottom left, altitude plot (fly vs Armstrong) bottom centre.
All values come from simulation logs; Armstrong's points from the ALSJ transcript.

Output: results/video/fly_lands_apollo11.mp4
"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import imageio.v2 as imageio  # noqa: E402
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollo_sim import constants as C  # noqa: E402
from apollo_sim import reference as ref  # noqa: E402
from apollo_sim.agc import AGC  # noqa: E402
from apollo_sim.brainview import BrainView  # noqa: E402
from apollo_sim.eyeview import draw_eye  # noqa: E402
from apollo_sim.flyeye import FlyEye  # noqa: E402
from apollo_sim.lm import at_ignition  # noqa: E402
from apollo_sim.render3d import Renderer  # noqa: E402
from apollo_sim.terrain import Terrain  # noqa: E402
from flybrain import Connectome  # noqa: E402

RUN = ROOT / "results" / "phase3c"
OUT = ROOT / "results" / "video"
FPS = 30
W, H = 1920, 1080
WARP_HIGH = 60.0   # PDI -> 1500 m: 2D profile (~9 s)
WARP_LOW = 10.0    # 1500 m -> P66: 3D, P64 approach (~7.5 s)
ALT_3D = 1500.0    # 3D view below this altitude
WARP_P66 = 1.5
# Channel mixing in the vision pipeline, measured in telemetry.npz (trial 0): forward flow leaks into the
# expansion channel so the GF fires -> hover; once forward speed drops, real approach is under-read -> fast descent.
HOVER_NOTE = (10.0, 32.0)   # P66+ s: h ~310 ft, vs ~0, GF 75-110 Hz, forward 20-38 ft/s
FAST_NOTE = (42.0, 58.0)    # P66+ s: sink 7-17 ft/s, GF 13-22 Hz
TITLE_S, END_S = 4.0, 8.0
GREEN, AMBER, RED, CYAN, WHITE, DIM = (120, 230, 140), (255, 190, 80), (240, 90, 80), (90, 210, 240), (235, 235, 240), (140, 145, 160)


def get_str(t):
    s = ref.PDI + t
    return f"{int(s // 3600):03d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


class Fonts:
    def __init__(self):
        f = pygame.font.SysFont
        self.title = f("segoeui", 64, bold=True)
        self.sub = f("segoeui", 28)
        self.h2 = f("segoeui", 26, bold=True)
        self.body = f("segoeui", 20)
        self.small = f("segoeui", 16)
        self.mono = f("consolas", 20)
        self.seg = f("consolas", 30, bold=True)


def text(surf, s, pos, font, col=WHITE, center=False, alpha=255):
    img = font.render(s, True, col)
    if alpha < 255:
        img.set_alpha(alpha)
    r = img.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    surf.blit(img, r)
    return r


def panel(surf, rect, a=170):
    p = pygame.Surface(rect[2:], pygame.SRCALPHA)
    p.fill((0, 0, 0, a))
    surf.blit(p, rect[:2])


def descent_track(terrain, t_handover):
    """PDI -> handover flown by the AGC (same start as trial 0)."""
    lm = at_ignition(terrain)
    agc = AGC(lm)
    rows, t = [], 0.0
    while t < t_handover:
        agc.update(t)
        lm.step(C_DT)
        t += C_DT
        rows.append((t, lm.s.pos[0], lm.datum_altitude, lm.altitude, lm.s.pitch, lm.s.thrust, lm.vertical_speed(),
                     lm.horizontal_speed(), lm.s.propellant, agc.program))
    return np.array(rows)


C_DT = 0.05
_STARS = np.random.default_rng(7).random((300, 3))


def draw_profile(screen, desc, k, F):
    """Side profile with lunar curvature: downrange (km) vs altitude (km, exaggerated)."""
    screen.set_clip((0, 0, 1280, 720))
    pygame.draw.rect(screen, (4, 5, 9), (0, 0, 1280, 720))
    for x, y, b in _STARS:
        cc = int(60 + 120 * b)
        screen.set_at((int(x * 1280), int(y * 560)), (cc, cc, cc))
    x_km = desc[:, 1] / 1000
    h_km = desc[:, 3] / 1000
    X0, X1 = x_km.min() - 10, 15
    sx = lambda xk: 60 + (xk - X0) / (X1 - X0) * 1160
    sy = lambda hk: 600 - hk / 16 * 480
    xs = np.linspace(X0, X1, 400)
    drop = (xs * 1000) ** 2 / (2 * C.R_MOON) / 1000
    pts = [(sx(a), sy(-d)) for a, d in zip(xs, drop)]
    pygame.draw.polygon(screen, (70, 70, 74), pts + [(1280, 720), (0, 720)])
    pygame.draw.lines(screen, (150, 150, 155), False, pts, 2)
    path = [(sx(a), sy(b)) for a, b in zip(x_km[: k + 1:20], h_km[: k + 1:20])]
    if len(path) > 1:
        pygame.draw.lines(screen, AMBER, False, path, 3)
    future = [(sx(a), sy(b)) for a, b in zip(x_km[k::40], h_km[k::40])]
    if len(future) > 1:
        pygame.draw.lines(screen, (90, 80, 50), False, future, 1)
    lx, ly = sx(x_km[k]), sy(h_km[k])
    pygame.draw.circle(screen, WHITE, (int(lx), int(ly)), 7)
    text(screen, "Eagle", (int(lx) + 10, int(ly) - 28), F.body)
    pygame.draw.line(screen, GREEN, (sx(0), sy(0) - 14), (sx(0), sy(0) + 2), 3)
    text(screen, "Tranquility Base", (int(sx(0)) - 70, int(sy(0)) + 8), F.small, GREEN)
    for hk in (5, 10, 15):
        text(screen, f"{hk} km", (454, int(sy(hk)) - 9), F.small, DIM)  # right of the telemetry panel
    text(screen, "side view, altitude exaggerated ~30x", (640, 700), F.small, DIM, True)
    screen.set_clip(None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pygame.init()
    F = Fonts()
    tel = np.load(RUN / "telemetry.npz")
    summ = json.loads((RUN / "summary.json").read_text())
    eye_imgs = np.load(RUN / "eye_trial0.npz")["images"].astype(np.float32)
    br = np.load(RUN / "brain_trial0.npz")
    t_h = float(tel["t_handover"])
    print("terrain...", flush=True)
    terrain = Terrain()
    ctx = moderngl.create_standalone_context()
    r3d = Renderer(terrain, 1280, 720, ctx=ctx)

    # brain colours by role
    c = Connectome()
    soma = np.load(ROOT / "data/processed/soma_um.npy")
    col = np.tile(np.array([1.0, 0.55, 0.15], np.float32), (c.n, 1))
    inputs = c.idx("LPLC2", "T4a", "T5a", "T4b", "T5b")
    outputs = c.idx("DNp01", "DNp26", "DNbe001", "DNp20", "DNg46", "DNp53", "DNge043")
    col[inputs] = (0.35, 0.85, 1.0)
    col[outputs] = (1.0, 0.25, 0.2)
    bv = BrainView(soma, col, 640, 1080, ctx=ctx)
    labels = {"LPLC2 (looming)": c.idx("LPLC2"), "T4/T5 (motion)": c.idx("T4a", "T5a"),
              "Giant Fiber -> ROD": c.idx("DNp01"), "brake DNs -> hand controller": c.idx("DNp26", "DNbe001", "DNp20")}

    print("PDI descent...", flush=True)
    desc = descent_track(terrain, t_h)
    eye = FlyEye(terrain)
    eye_prev = None

    # P66 log (trial 0)
    tt = tel["t"] - t_h
    alive = ~tel["landed"][:, 0]
    n_tick = int(alive.sum()) + 1
    x0, h0, p0 = tel["x"][:, 0], tel["h"][:, 0], tel["pitch"][:, 0]
    fuel = tel["propellant"][:, 0]
    thrust = np.r_[0.0, -np.diff(fuel) / 0.2 * C.DPS_ISP * C.G0]
    s0 = next(s for s in summ["trials"] if s["trial"] == "fly_0")
    p66_len = float(s0["p66_duration_s"])
    co = [cc for cc in ref.callouts_si() if cc[0] >= ref.EVENTS["P66"] - ref.PDI - 10]
    co_t = [cc[0] - (ref.EVENTS["P66"] - ref.PDI) for cc in co if cc[1]]
    co_h = [cc[1] / C.FT for cc in co if cc[1]]

    writer = imageio.get_writer(OUT / "fly_lands_apollo11.mp4", fps=FPS, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    screen = pygame.Surface((W, H))
    n_title = int(TITLE_S * FPS)
    t_3d = float(desc[np.argmax(desc[:, 3] <= ALT_3D), 0])
    s_high = t_3d / WARP_HIGH
    n_desc = int((s_high + (desc[-1, 0] - t_3d) / WARP_LOW) * FPS)
    n_p66 = int(p66_len / WARP_P66 * FPS) + 1
    n_end = int(END_S * FPS)
    total = n_title + n_desc + n_p66 + n_end
    hist_t, hist_h = [], []
    print(f"frames: {total}", flush=True)

    for f in range(total):
        screen.fill((0, 0, 0))
        phase = "title" if f < n_title else "desc" if f < n_title + n_desc else "p66" if f < n_title + n_desc + n_p66 else "end"
        rates = np.zeros(c.n)
        if phase in ("title", "desc"):
            s_vid = (f - n_title) / FPS
            warp = WARP_HIGH if s_vid < s_high else WARP_LOW
            t_d = s_vid * WARP_HIGH if s_vid < s_high else t_3d + (s_vid - s_high) * WARP_LOW
            k = 0 if phase == "title" else min(int(t_d / C_DT), len(desc) - 1)
            t_sim, x, y, alt, pitch, thr, vs, hs, fu, prog = desc[k]
            tilt_cmd, cam = pitch, "chase"
            t_p66 = None
        else:
            t_p66 = min((f - n_title - n_desc) / FPS * WARP_P66, p66_len)
            q = min(t_p66 / 0.2, n_tick - 1.001)
            i = int(q)
            w = q - i
            j = min(i + 1, len(x0) - 1)
            x = (1 - w) * x0[i] + w * x0[j]
            alt = max((1 - w) * h0[i] + w * h0[j], 0.0)
            y = alt + float(terrain.height(x))
            pitch = (1 - w) * p0[i] + w * p0[j]
            thr = thrust[j] if phase == "p66" else 0.0
            vs, hs, fu = tel["vs"][j, 0], tel["hs"][j, 0], fuel[j]
            prog, t_sim = 66, t_h + t_p66
            cam = "window" if 40 <= t_p66 < 75 else "chase"
            s, e = br["ptr"][i], br["ptr"][i + 1]
            if phase == "p66":
                rates[br["idx"][s:e]] = br["cnt"][s:e] / 0.2
        # 3D view (2D profile when high)
        if phase in ("title", "desc") and alt > ALT_3D:
            draw_profile(screen, desc, k, F)
        else:
            img3d = r3d.render(x, y, pitch, thr / C.DPS_NOMINAL_MAX, cam, dt=1 / FPS * (WARP_P66 if t_p66 is not None else 1.0))
            screen.blit(pygame.surfarray.make_surface(np.ascontiguousarray(img3d.transpose(1, 0, 2))), (0, 0))
        # brain
        bv.update(rates, 1 / FPS * (WARP_P66 if phase == "p66" else 1.0))
        yaw = 0.25 * math.sin(f / FPS * 0.25)
        imgb = bv.render(yaw)
        screen.blit(pygame.surfarray.make_surface(np.ascontiguousarray(imgb.transpose(1, 0, 2))), (1280, 0))
        text(screen, "Fruit fly brain - live spikes", (1300, 16), F.h2)
        text(screen, "MaleCNS v1.0 connectome, 165,122 neurons", (1300, 48), F.small, DIM)
        text(screen, "each dot = one neuron at its real cell-body position", (1300, 68), F.small, DIM)
        for k_, (lab, idx) in enumerate(labels.items()):
            P = soma[idx]
            P = P[np.isfinite(P[:, 0])]
            if len(P) == 0:
                continue
            px = bv.project(P, yaw)
            colr = CYAN if k_ < 2 else RED
            left = px[:, 0] < 320
            for side in (left, ~left):
                if side.sum():
                    cx, cy = px[side].mean(0)
                    pygame.draw.circle(screen, colr, (int(1280 + cx), int(cy)), 10 + 3 * (k_ % 2), 1)
        panel(screen, (1290, 940, 620, 100), 160)
        text(screen, "circles: input neurons  -  LPLC2 (looming), T4/T5 (motion)", (1300, 950), F.small, CYAN)
        text(screen, "circles: output neurons  -  Giant Fiber -> descent-rate switch,", (1300, 975), F.small, RED)
        text(screen, "brake descending neurons -> hand controller", (1300, 997), F.small, RED)
        if phase in ("title", "desc"):
            text(screen, "standing by - the computer is flying", (1600, 1062), F.small, DIM, center=True)

        # DSKY
        panel(screen, (1040, 12, 228, 210), 190)
        text(screen, "PROG", (1170, 18), F.small, DIM)
        text(screen, f"{int(prog):02d}", (1170, 34), F.seg, GREEN)
        noun = 63 if prog in (63, 64) else 60
        text(screen, "VERB 06", (1056, 40), F.small, DIM)
        text(screen, f"NOUN {noun}", (1056, 58), F.small, DIM)
        r1 = math.hypot(vs, hs) / C.FT if noun == 63 else hs / C.FT
        for k_, (lab, val) in enumerate([("R1", f"{r1:+07.1f}"), ("R2", f"{vs / C.FT:+07.1f}"), ("R3", f"{alt / C.FT:+07.0f}")]):
            text(screen, lab, (1056, 98 + 38 * k_), F.small, DIM)
            text(screen, val, (1100, 88 + 38 * k_), F.seg, GREEN)
        # telemetry
        panel(screen, (12, 12, 430, 150), 200)
        text(screen, f"GET {get_str(t_sim)}", (24, 18), F.mono, WHITE)
        text(screen, f"altitude   {alt / C.FT:7.0f} ft", (24, 46), F.mono)
        text(screen, f"descent    {-vs / C.FT:7.1f} ft/s", (24, 72), F.mono)
        text(screen, f"forward    {hs / C.FT:7.1f} ft/s", (24, 98), F.mono)
        fuel_pct = fu / C.PROPELLANT_PDI * 100
        text(screen, f"fuel       {fuel_pct:7.1f} %", (24, 124), F.mono, RED if fuel_pct < 5.6 else WHITE)
        if cam == "window":
            panel(screen, (0, 688, 1280, 26), 150)
            text(screen, "view: out of Aldrin's window (like the 16 mm film camera)", (640, 700), F.small, DIM, center=True)

        # fly's eye
        if t_p66 is not None and phase == "p66":
            img_e = eye_imgs[min(i, len(eye_imgs) - 1)]
            if eye_prev is None or i != eye_prev[0]:
                _, _, _, prog_m, m_el = eye.motion(img_e, 0.2)
                eye_prev = (i, prog_m, m_el)
            draw_eye(screen, (14, 754, 650, 290), img_e, eye_prev[1], eye_prev[2], arrows_every=4)
            text(screen, "What the fly sees - 1296 ommatidia, lower half of both eyes", (14, 728), F.body)
            text(screen, "orange = motion detected (Reichardt detectors, as in T4/T5)", (14, 1061), F.small, DIM)
        elif phase == "end" and eye_prev is not None:
            draw_eye(screen, (14, 754, 650, 290), eye_imgs[min(eye_prev[0], len(eye_imgs) - 1)], arrows_every=4)
            text(screen, "What the fly sees - on the surface", (14, 728), F.body)
        else:
            text(screen, "What the fly sees", (14, 728), F.body, DIM)
            text(screen, f"(the fly takes over in P66 at {h0[0] / C.FT:.0f} ft, near where Armstrong did)", (14, 760), F.small, DIM)

        # altitude plot
        gx, gy, gw, gh = 700, 742, 560, 300
        pygame.draw.rect(screen, (12, 12, 16), (gx - 10, gy - 14, gw + 20, gh + 50))
        text(screen, "P66 altitude: fly vs Neil Armstrong (Apollo 11)", (gx, gy - 12), F.body)
        sx = lambda t: gx + 40 + t / 140 * (gw - 50)
        sy = lambda hf: gy + gh - 10 - hf / 520 * (gh - 40)
        pygame.draw.line(screen, DIM, (sx(0), sy(0)), (sx(140), sy(0)), 1)
        pygame.draw.line(screen, DIM, (sx(0), sy(0)), (sx(0), sy(500)), 1)
        for hv in (100, 200, 300, 400, 500):
            text(screen, f"{hv}", (gx, sy(hv) - 9), F.small, DIM)
        for tv in (0, 30, 60, 90, 120):
            text(screen, f"{tv}s", (sx(tv) - 10, sy(0) + 4), F.small, DIM)
        pts = [(sx(a), sy(b)) for a, b in zip(co_t, co_h) if a <= 140]
        if len(pts) > 1:
            pygame.draw.lines(screen, RED, False, pts, 2)
            for p in pts:
                pygame.draw.circle(screen, RED, (int(p[0]), int(p[1])), 3)
        if t_p66 is not None:
            hist_t.append(t_p66)
            hist_h.append(alt / C.FT)
            if len(hist_t) > 1:
                pygame.draw.lines(screen, GREEN, False, [(sx(a), sy(b)) for a, b in zip(hist_t, hist_h)], 3)
        text(screen, "fly", (gx + gw - 120, gy + 20), F.body, GREEN)
        text(screen, "Armstrong", (gx + gw - 120, gy + 46), F.body, RED)

        # title / banners
        if phase == "title":
            a = int(255 * min(1, f / 15, (n_title - f) / 15))
            panel(screen, (0, 250, 1280, 230), min(a, 200))
            text(screen, "Can a fruit fly brain land Apollo 11?", (640, 320), F.title, WHITE, True, a)
            text(screen, "165,122-neuron connectome  |  Apollo 11 guidance equations  |  Tranquility Base", (640, 400), F.sub, DIM, True, a)
        elif phase == "desc":
            panel(screen, (0, 640, 1280, 46), 150)
            text(screen, f"Powered descent flown by the Apollo Guidance Computer (Luminary 099 guidance law)  -  {warp:.0f}x speed",
                 (640, 663), F.body, WHITE, True)
        elif phase == "p66":
            if t_p66 < 6:
                a = int(255 * min(1, (6 - t_p66) / 1.5))
                panel(screen, (0, 250, 1280, 170), min(a, 200))
                text(screen, "102:43:22 - here Armstrong took manual control (P66)", (640, 300), F.sub, WHITE, True, a)
                text(screen, "This time, a fruit fly's brain is at the controls.", (640, 350), F.h2, AMBER, True, a)
            for (n0, n1), lines in ((HOVER_NOTE, ("Model note: forward motion leaks into the fly's looming estimate in this model,",
                                                  "so the Giant Fiber fires as if the ground were approaching -> hover.")),
                                    (FAST_NOTE, ("Model note: as forward speed drops, the looming estimate under-reads",
                                                 "the real approach rate -> faster descent."))):
                if n0 <= t_p66 < n1:
                    a = int(255 * min(1, (t_p66 - n0) / 1.0, (n1 - t_p66) / 1.0))
                    panel(screen, (240, 572, 800, 58), min(a, 190))
                    text(screen, lines[0], (640, 588), F.small, AMBER, True, a)
                    text(screen, lines[1], (640, 611), F.small, AMBER, True, a)
            panel(screen, (0, 640, 1280, 46), 150)
            text(screen, f"P66  -  looming -> Giant Fiber -> descent-rate switch   |   optic flow -> descending neurons -> hand controller   ({WARP_P66:g}x)",
                 (640, 663), F.small, WHITE, True)
        else:
            panel(screen, (0, 150, 1280, 470), 210)
            text(screen, "THE EAGLE HAS LANDED", (640, 210), F.title, GREEN, True)
            text(screen, "...flown through P66 by a simulated fruit fly brain", (640, 275), F.sub, WHITE, True)
            rows = [("", "fruit fly brain", "Neil Armstrong"),
                    ("P66 duration", f"{p66_len:.0f} s", f"{ref.EVENTS['contact_light'] - ref.EVENTS['P66']} s"),
                    ("sideways speed at touchdown", f"{abs(s0['touchdown_lateral_mps']) / C.FT:.1f} ft/s", f"{ref.TOUCHDOWN_LATERAL / C.FT:.1f} ft/s"),
                    ("descent rate at contact light", f"{s0['contact_light_speed_mps'] / C.FT:.1f} ft/s", "~0.5 ft/s (callout at 20 ft)"),
                    ("fuel left", f"{s0['fuel_left_kg']:.0f} kg", f"~{ref.PROPELLANT_LEFT:.0f} kg")]
            for k_, (a_, b_, c_) in enumerate(rows):
                yy = 330 + 38 * k_
                text(screen, a_, (190, yy), F.body, DIM)
                text(screen, b_, (640, yy), F.h2 if k_ == 0 else F.body, GREEN, True)
                text(screen, c_, (930, yy), F.h2 if k_ == 0 else F.body, RED, True)
            text(screen, "Simulation. Brain: leaky integrate-and-fire model (Shiu et al. 2024) on MaleCNS v1.0 (Janelia/Google 2026),",
                 (640, 545), F.small, DIM, True)
            text(screen, "calibrated to FlyWire. Vision: Reichardt motion detectors. Encoder/decoder gains and readout neurons are design choices.",
                 (640, 567), F.small, DIM, True)
            text(screen, "Armstrong data: Apollo Lunar Surface Journal (Aldrin's callouts).", (640, 589), F.small, DIM, True)

        frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        writer.append_data(frame)
        if f % 300 == 0:
            print(f"{f}/{total}", flush=True)
    writer.close()
    print("saved:", OUT / "fly_lands_apollo11.mp4")


if __name__ == "__main__":
    main()
