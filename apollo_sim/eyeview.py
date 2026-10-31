"""Draws the fly's-eye image as a honeycomb of ommatidia on a pygame surface.

Horizontal axis is azimuth (-180 left-rear ... 0 front ... +180 right-rear), vertical is elevation
(0 horizon at the top, -90 straight down at the bottom). Alternate rows are offset by half a cell.
Optionally overlays the local Reichardt motion signals as arrows.
"""
import math

import numpy as np
import pygame


def _hex(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * k + 30)), cy + r * math.sin(math.radians(60 * k + 30))) for k in range(6)]


def draw_eye(surf, rect, img, prog=None, m_el=None, gamma=0.8, arrows_every=3):
    """img: (n_el, n_az) luminance [0..~1.5]; rect: (x, y, w, h)."""
    x0, y0, w, h = rect
    n_el, n_az = img.shape
    cw = w / (n_az + 0.5)
    ch = h / n_el
    r = min(cw / math.sqrt(3) * 1.02, ch / 1.5 * 1.02)
    pygame.draw.rect(surf, (8, 8, 10), rect)
    ref = max(float(np.percentile(img, 97)), 1e-3)  # auto exposure, like photoreceptor adaptation
    v = np.clip(img / ref, 0, 1) ** gamma
    for i in range(n_el):  # i=0 is the lowest row (-87.5 deg)
        cy = y0 + h - (i + 0.5) * ch
        off = 0.5 * cw if i % 2 else 0.0
        for j in range(n_az):
            cx = x0 + off + (j + 0.5) * cw
            c = int(255 * v[i, j])
            pygame.draw.polygon(surf, (c, c, int(c * 0.95)), _hex(cx, cy, r))
    if prog is not None and m_el is not None:
        scale = 0.5 * cw / (np.percentile(np.abs(prog), 95) + 1e-9)
        for i in range(0, n_el, arrows_every):
            cy = y0 + h - (i + 0.5) * ch
            for j in range(0, n_az, arrows_every):
                cx = x0 + (j + 0.5) * cw
                dx = float(np.clip(prog[i, j] * scale, -cw * 1.5, cw * 1.5)) * (1 if (j + 0.5) * 5 - 180 > 0 else -1)
                dy = -float(np.clip(m_el[i, j] * scale, -ch * 1.5, ch * 1.5))
                if abs(dx) + abs(dy) > 1.5:
                    pygame.draw.line(surf, (255, 170, 60), (cx, cy), (cx + dx, cy + dy), 2)
    font = pygame.font.SysFont("consolas", 12)
    for az in (-180, -90, 0, 90, 180):
        px = x0 + (az + 180) / 360 * w
        surf.blit(font.render(f"{az}°", True, (150, 150, 160)), (px - 12, y0 + h + 2))
