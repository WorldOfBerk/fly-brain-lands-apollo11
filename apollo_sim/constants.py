"""Moon, Apollo 11 LM and Luminary constants, each with its source.

Units are SI (m, s, kg, N); AGC scaled units (m/cs, m/cs^2, ...) are converted here.
"""
import math

FT = 0.3048
LBF = 4.448222
LB = 0.45359237

# --- Moon ---
MU_MOON = 4.9028e12  # m^3/s^2
R_MOON = 1_737_400.0  # m, mean radius

# --- Apollo 11 LM (Eagle) ---
# Mass at PDI 33,297 lb (Apollo 11 Mission Report); ~770 lb propellant left at touchdown, ~100 lb unusable (ALSJ)
MASS_PDI = 33_297 * LB  # 15 103 kg
PROPELLANT_PDI = 8_150.0  # kg, DPS propellant (18,184 lb loaded, ~100 kg used at DOI)
UNUSABLE_PROPELLANT = 100 * LB  # kg
# Low-level sensor trips at 5.6 % (ALSJ 102:44:31); bingo ~ low level + 114 s (Apollo 11 rule)
LOW_LEVEL_FRACTION = 0.056
BINGO_AFTER_LOW_LEVEL = 114.0  # s, at average consumption

# --- DPS (Descent Propulsion System) ---
DPS_NOMINAL_MAX = 10_500 * LBF  # 46,706 N, throttle percentages refer to this
DPS_FTP = 9_870 * LBF  # Fixed Throttle Position (92.5 %), 43,904 N
DPS_THROTTLE_MIN = 0.10
DPS_THROTTLE_MAX = 0.65  # no throttling above this, only FTP
DPS_ISP = 305.0  # s (vacuum ~311 s, lower when throttled)
G0 = 9.80665

# --- RCS / attitude ---
# Pitch inertia scales with mass: ~31,000 kg m^2 at PDI, ~15,000 kg m^2 at touchdown.
PITCH_INERTIA_PER_KG = 2.05  # m^2
RCS_PITCH_TORQUE = 2 * 445.0 * 1.68  # 2 jets x 100 lbf x moment arm ~ 1,500 N m
ATT_MAX_RATE = math.radians(5.0)  # rad/s, auto maneuver rate limit

# --- Luminary099 constants (virtualagc/Luminary099) ---
THROTLAG = 0.2  # s, CONTROLLED_CONSTANTS.agc:134
HIGHESTF = 4.34546769e4  # N, LUNAR_LANDING_GUIDANCE_EQUATIONS.agc:1476 (AFC limit)
GUIDANCE_PERIOD = 2.0  # s, SERVICER cycle
ZOOMTIME = 26.0  # s, throttle-up after ignition at 10 % (ALSJ 102:33:25 "24, 25, 26, throttle up")
RODSCALE = 1 * FT  # m/s per ROD click (LUNAR_LANDING_GUIDANCE_EQUATIONS.agc: RODCOMP)
TAUROD = 1.5  # s, P66 rate loop time constant

# --- Targets (pad load) ---
# Source: virtualagc/LUM69R2/PADLOADS.agc (Luminary 069, Apollo 10 LM). Apollo 11 flew the same two-phase
# descent design; a Luminary099 pad load list is not in the repository.
# Guidance frame: x = up at the landing site, z = downrange (positive towards the site).
# m/cs -> x100 m/s, m/cs^2 -> x1e4 m/s^2, m/cs^3 -> x1e6 m/s^3
BRAKING = dict(  # P63 target: "high gate"
    R=(2923.62643, -10083.9629),
    V=(-0.483907728 * 100, 1.71785605 * 100),
    A=(-5.22722473e-5 * 1e4, -2.86621213e-4 * 1e4),
)
APPROACH = dict(  # P64 target: "low gate"
    R=(23.5092239, -0.528319999),
    V=(-9.44879999e-3 * 100, 3.96239999e-3 * 100),
    A=(1.52399999e-6 * 1e4, -1.98119999e-5 * 1e4),
)
TENDBRAK = 20.0  # s, P63 -> P64 switch criterion
TENDAPPR = 10.0  # s, P64 -> P65 switch criterion
LOWCRIT = 0.57  # fraction of max thrust: below this, go to throttleable
HIGHCRIT = 0.63  # above this, go to FTP
# Ignition point (guidance frame)
RIGN = (-40943.2231, -440014.934)  # m
VIGN = 1699.52182  # m/s


def _jerk_through(target, z0, vz0):
    """Downrange jerk J (and TTF at that point, T < 0) such that the reference trajectory
    r(T) = R + V T + A T^2/2 + J T^3/6 passes through (z0, vz0).

    The AGC scaling of JBRFG*/JAPFG* in the pad load could not be converted to SI reliably, so J is solved
    to be consistent with the other targets of the same pad load. Check: the approach phase gives
    ~0.011 m/s^3, in line with NASSP's Apollo 12 value
    (Config/ProjectApollo/RTCC/Apollo 12 Constants.txt, PDI_JARFGZ = 0.0377 ft/s^3 = 0.0115 m/s^3).
    """
    R, V, A = target["R"][1], target["V"][1], target["A"][1]
    lo, hi = -2000.0, -1.0
    f = lambda T: R + V * T + A * T**2 / 2 + (2 * (vz0 - V - A * T) / T**2) * T**3 / 6 - z0
    for _ in range(200):  # bisection
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    T = (lo + hi) / 2
    return 2 * (vz0 - V - A * T) / T**2, T


BRAKING["J"], TTF_IGN = _jerk_through(BRAKING, RIGN[1], VIGN)
APPROACH["J"], TTF_P64 = _jerk_through(APPROACH, BRAKING["R"][1], BRAKING["V"][1])
