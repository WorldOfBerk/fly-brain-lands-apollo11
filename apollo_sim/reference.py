"""The real Apollo 11 landing: Aldrin's DSKY callouts and event times.

Source: Apollo Lunar Surface Journal, "The First Lunar Landing"
https://apollojournals.org/alsj/a11/a11.landing.html
Times are GET. Ignition at 102:33:07 (ALSJ: "actual PDI occurred at 102:33:07").
Altitude ft, descent rate ft/s (down positive), forward speed ft/s. None where not called out.
"""
from .constants import FT


def get(h, m, s):
    return h * 3600 + m * 60 + s


PDI = get(102, 33, 7)

# (GET, altitude ft, descent rate ft/s, forward ft/s, callout)
CALLOUTS = [
    (get(102, 41, 51), 5000, 100, None, "5000, 100 ft/s descent"),
    (get(102, 42, 13), 3000, 70, None, "3000 at 70"),
    (get(102, 42, 24), 2000, 50, None, "2000 at 50"),
    (get(102, 42, 37), 1000, 30, None, "1000 at 30"),
    (get(102, 43, 1), 750, 23, None, "750, coming down at 23"),
    (get(102, 43, 7), 700, 21, None, "700 feet, 21 down"),
    (get(102, 43, 11), 600, 19, None, "600 feet, down at 19"),
    (get(102, 43, 16), 540, 15, None, "540 feet, down at 15"),
    (get(102, 43, 26), 400, 9, 58, "400 feet, down at 9, 58 forward"),
    (get(102, 43, 33), 350, 4, None, "350 feet, down at 4"),
    (get(102, 43, 35), 330, 3.5, None, "330, three and a half down"),
    (get(102, 43, 46), 300, 3.5, 47, "300 feet, down 3 1/2, 47 forward"),
    (get(102, 43, 52), 270, 1.5, None, "1 1/2 down, 270"),
    (get(102, 44, 7), 250, 2.5, 19, "250, down at 2 1/2, 19 forward"),
    (get(102, 44, 16), 220, 3.5, 13, "3 1/2 down, 220 feet, 13 forward"),
    (get(102, 44, 24), 200, 4.5, 11, "200 feet, 4 1/2 down"),
    (get(102, 44, 31), 160, 6.5, None, "160 feet, 6 1/2 down"),
    (get(102, 44, 33), None, 5.5, 9, "5 1/2 down, 9 forward"),
    (get(102, 44, 40), 120, None, None, "120 feet"),
    (get(102, 44, 45), 100, 3.5, 9, "100 feet, 3 1/2 down, 9 forward"),
    (get(102, 44, 54), 75, 0.5, 6, "75 feet, down a half, 6 forward"),
    (get(102, 45, 17), 40, 2.5, None, "40 feet, down 2 1/2"),
    (get(102, 45, 21), 30, 2.5, None, "30 feet, 2 1/2 down"),
    (get(102, 45, 25), 20, 0.5, 4, "20 feet, down a half, 4 forward"),
]

EVENTS = {
    "PDI": PDI,
    "throttle_down": PDI + 6 * 60 + 25,  # Houston 102:39:01: "6 plus 25, throttle down" (expected time)
    "P64": get(102, 41, 35),
    "P66": get(102, 43, 22),  # Armstrong takes manual control
    "contact_light": get(102, 45, 40),
    "shutdown": get(102, 45, 43),
}
TOUCHDOWN_VERTICAL = 1.7 * FT  # m/s (Mission Report)
TOUCHDOWN_LATERAL = 2.1 * FT  # m/s
PROPELLANT_LEFT = 770 * 0.45359237  # kg (~100 lb of it unusable)


def callouts_si():
    """(s from PDI, altitude m, vertical speed m/s (up +), forward m/s, callout)."""
    out = []
    for t, h, vd, vf, txt in CALLOUTS:
        out.append((t - PDI, h * FT if h is not None else None, -vd * FT if vd is not None else None,
                    vf * FT if vf is not None else None, txt))
    return out
