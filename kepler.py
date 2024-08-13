# Approximate orbits
from scipy.optimize import newton
from datetime import datetime, timezone
import numpy as np
from time import time

def gregorian_to_julian_day_number(month, day, year):
    """Convert the given proleptic Gregorian date to the equivalent Julian Day Number."""
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12, inclusive")
    if day < 1 or day > 31:
        raise ValueError("day must be between 1 and 31, inclusive")
    A = int((month - 14) / 12)
    B = 1461 * (year + 4800 + A)
    C = 367 * (month - 2 - 12 * A)
    E = int((year + 4900 + A) / 100)
    JDN = int(B / 4) + int(C / 12) - int(3 * E / 4) + day - 32075
    return JDN


def gregorian_to_julian_date(dt):
    """Convert a Gregorian date to a Julian Date."""
    JDN = gregorian_to_julian_day_number(dt.month, dt.day, dt.year)
    JDT = JDN + (dt.hour - 12) / 24 + dt.minute / 1_440 + dt.second / 86_400 + dt.microsecond / 86_400_000_000
    return JDT

def kepler(E, M_e, e):
    """Kepler's equation, to be used in a Newton solver."""
    return E - e * np.sin(E) - M_e

def d_kepler_d_E(E, M_e, e):
    """The derivative of Kepler's equation, to be used in a Newton solver.
    
    Note that the argument M_e is unused, but must be present so the function
    arguments are consistent with the kepler function.
    """
    return 1 - e * np.cos(E)

start = time()
mercury = {
    "a": 0.38709927, "a_dot": 0.00000037,
    "e": 0.20563593, "e_dot": 0.00001906,
    "i": 7.00497902, "i_dot": -0.0059475,
    "L": 252.250324, "L_dot": 149472.674,
    "long_peri": 77.4577963, "long_peri_dot": 0.16047689,
    "long_node":48.3307659, "long_node_dot": -0.1253408,
}

T_eph = datetime(month=8, day=6, year=2024, hour=19, minute=30, tzinfo=timezone.utc)
JDT = gregorian_to_julian_date(T_eph)

T = (JDT - 2_451_545) / 36_525
a = (mercury["a"] + mercury["a_dot"] * T) * 149_597_870.7
e = mercury["e"] + mercury["e_dot"] * T
i = np.radians(mercury["i"] + mercury["i_dot"] * T)
L = np.radians(mercury["L"] + mercury["L_dot"] * T)
long_peri = np.radians(mercury["long_peri"] + mercury["long_peri_dot"] * T)
long_node = np.radians(mercury["long_node"] + mercury["long_node_dot"] * T)

M_e = (L - long_peri) % (2 * np.pi)

E = newton(func=kepler, fprime=d_kepler_d_E, x0=np.pi, args=(M_e, e))
nu = (2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))) % (2 * np.pi)

omega = long_peri - long_node

r = a*(1-e**2)/((1+e*np.cos(nu)))

print(f"𝑎 = {a:.5G} km", f"𝑒 = {e:.5F}", f"𝑖 = {np.degrees(i):.2F}°", f"𝜃 = {np.degrees(nu):.2F}°",
      f"𝜔 = {np.degrees(omega):.2F}°", f"𝛺 = {np.degrees(long_node):.2F}°", f"r = {r:.2F} km", sep="\n")

print(f'Run time:{round(time() - start, 2)}')

from distance import Distance
