"""
Analog component library — simplified BSIM-like parameters at 130nm.

Provides per-component reference parameters (W, L, Vth, transit caps), the
six-class device taxonomy used in node features, and tiny helpers to estimate
small-signal quantities (gm, Cgs) from a device's geometry and bias current.

This is deliberately simplified — real BSIM has hundreds of parameters per
device. The intent is to produce plausible per-node feature values and
deterministic ground-truth labels for the analog tasks.
"""

import math

# Six analog device classes used in the device-type one-hot feature.
DEVICE_FUNCTIONS = ["NMOS", "PMOS", "R", "C", "CURRENT_MIRROR", "DIFF_PAIR"]
DEVICE_TO_IDX = {n: i for i, n in enumerate(DEVICE_FUNCTIONS)}
NUM_DEVICE_TYPES = len(DEVICE_FUNCTIONS)


# 130nm process constants (order-of-magnitude reasonable).
# Units: m, F/m^2, m^2/(V·s), V.
PROCESS = {
    "L_min": 130e-9,
    "Vdd": 1.2,
    "Cox": 6e-3,          # F/m^2
    "mu_n": 300e-4,       # m^2/(V·s)
    "mu_p": 100e-4,       # m^2/(V·s)
    "Vth_n": 0.4,
    "Vth_p": -0.4,
    "lambda_n": 0.1,      # 1/V (channel-length modulation)
    "lambda_p": 0.15,
}


# Default parameter sheets per device class. Subcircuit primitives
# (current_mirror, diff_pair) carry meta information rather than physics.
DEFAULT_DEVICE_PARAMS = {
    "NMOS": {
        "W": 2e-6, "L": 130e-9, "Vth": PROCESS["Vth_n"],
        "Cgs": 1.0e-15, "Cgd": 0.2e-15, "gm": 1e-3, "gds": 1e-5,
        "intrinsic_delay_ps": 25.0,
        "output_resistance": 1e5,
        "input_capacitance": 1.2e-15,
    },
    "PMOS": {
        "W": 5e-6, "L": 130e-9, "Vth": PROCESS["Vth_p"],
        "Cgs": 2.5e-15, "Cgd": 0.4e-15, "gm": 0.7e-3, "gds": 8e-6,
        "intrinsic_delay_ps": 40.0,
        "output_resistance": 1.25e5,
        "input_capacitance": 2.9e-15,
    },
    "R": {
        "R": 10e3, "parasitic_C": 0.05e-15,
        "intrinsic_delay_ps": 5.0, "output_resistance": 10e3,
        "input_capacitance": 0.05e-15,
    },
    "C": {
        "C": 100e-15, "parasitic_R": 5.0,
        "intrinsic_delay_ps": 1.0, "output_resistance": 1e9,
        "input_capacitance": 100e-15,
    },
    "CURRENT_MIRROR": {
        "intrinsic_delay_ps": 30.0, "output_resistance": 1e5,
        "input_capacitance": 1.5e-15,
        # Composite primitive; the actual W/L lives on the constituent
        # NMOS/PMOS nodes inside the subcircuit.
    },
    "DIFF_PAIR": {
        "intrinsic_delay_ps": 35.0, "output_resistance": 1.2e5,
        "input_capacitance": 1.8e-15,
    },
}


def device_params(device_type: str) -> dict:
    """Return a copy of the default parameter sheet for a device type."""
    return dict(DEFAULT_DEVICE_PARAMS.get(device_type, {}))


def estimate_gm(device_type: str, W: float, L: float, ID: float = 50e-6) -> float:
    """Square-law gm estimate for MOS, 0 for passives.

    gm = sqrt(2 * mu * Cox * (W/L) * ID).
    """
    if device_type == "NMOS":
        mu = PROCESS["mu_n"]
    elif device_type == "PMOS":
        mu = PROCESS["mu_p"]
    else:
        return 0.0
    return math.sqrt(2.0 * mu * PROCESS["Cox"] * (W / max(L, 1e-12)) * max(ID, 1e-9))


def estimate_cgs(device_type: str, W: float, L: float) -> float:
    """(2/3) * Cox * W * L for MOS; 0 otherwise."""
    if device_type in ("NMOS", "PMOS"):
        return (2.0 / 3.0) * PROCESS["Cox"] * W * L
    return 0.0


def estimate_intrinsic_gain(device_type: str, W: float, L: float, ID: float = 50e-6) -> float:
    """gm * ro estimate: gm/gds where gds ~ lambda * ID."""
    if device_type not in ("NMOS", "PMOS"):
        return 0.0
    lam = PROCESS["lambda_n"] if device_type == "NMOS" else PROCESS["lambda_p"]
    gm = estimate_gm(device_type, W, L, ID)
    gds = lam * max(ID, 1e-9)
    return gm / max(gds, 1e-12)
