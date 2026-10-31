"""The AGC flying alone from PDI to touchdown (P63 -> P64 -> P65). Returns a log."""
import numpy as np

from .agc import AGC
from .lm import at_ignition

DT = 0.05


def fly(max_t=1000.0, on_step=None):
    lm = at_ignition()
    agc = AGC(lm)
    rows = []
    t = 0.0
    while not lm.s.landed and t < max_t:
        agc.update(t)
        if on_step:
            on_step(t, lm, agc)
        lm.step(DT)
        t += DT
        if len(rows) == 0 or t - rows[-1][0] >= 0.5:
            rows.append((t, lm.altitude, lm.vertical_speed(), lm.horizontal_speed(), lm.s.thrust,
                         lm.s.propellant, np.degrees(lm.s.pitch), agc.program, agc.ttf))
    return lm, agc, np.array(rows)
