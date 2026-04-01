"""
Nangate45 Open Cell Library -- timing and capacitance parameters.

Values derived from the NanGate 45nm Open Cell Library (FreePDK45).
Each cell entry contains:
  - intrinsic_delay (ns): gate propagation delay at nominal load
  - input_cap (fF): input pin capacitance
  - output_res (ohm): effective output resistance (for load-dependent delay)
  - drive_strength (relative): drive capability
  - num_inputs: number of input pins
"""

NANGATE45_CELLS = {
    "INV_X1": {
        "intrinsic_delay": 0.017,
        "input_cap": 0.98,
        "output_res": 320.0,
        "drive_strength": 1.0,
        "num_inputs": 1,
        "function": "NOT",
    },
    "INV_X2": {
        "intrinsic_delay": 0.014,
        "input_cap": 1.96,
        "output_res": 160.0,
        "drive_strength": 2.0,
        "num_inputs": 1,
        "function": "NOT",
    },
    "BUF_X1": {
        "intrinsic_delay": 0.032,
        "input_cap": 0.98,
        "output_res": 320.0,
        "drive_strength": 1.0,
        "num_inputs": 1,
        "function": "BUF",
    },
    "BUF_X2": {
        "intrinsic_delay": 0.027,
        "input_cap": 1.96,
        "output_res": 160.0,
        "drive_strength": 2.0,
        "num_inputs": 1,
        "function": "BUF",
    },
    "NAND2_X1": {
        "intrinsic_delay": 0.024,
        "input_cap": 1.05,
        "output_res": 350.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "NAND",
    },
    "NAND2_X2": {
        "intrinsic_delay": 0.020,
        "input_cap": 2.10,
        "output_res": 175.0,
        "drive_strength": 2.0,
        "num_inputs": 2,
        "function": "NAND",
    },
    "NAND3_X1": {
        "intrinsic_delay": 0.031,
        "input_cap": 1.05,
        "output_res": 420.0,
        "drive_strength": 1.0,
        "num_inputs": 3,
        "function": "NAND",
    },
    "NOR2_X1": {
        "intrinsic_delay": 0.028,
        "input_cap": 1.10,
        "output_res": 380.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "NOR",
    },
    "NOR2_X2": {
        "intrinsic_delay": 0.023,
        "input_cap": 2.20,
        "output_res": 190.0,
        "drive_strength": 2.0,
        "num_inputs": 2,
        "function": "NOR",
    },
    "NOR3_X1": {
        "intrinsic_delay": 0.038,
        "input_cap": 1.10,
        "output_res": 460.0,
        "drive_strength": 1.0,
        "num_inputs": 3,
        "function": "NOR",
    },
    "AND2_X1": {
        "intrinsic_delay": 0.038,
        "input_cap": 1.05,
        "output_res": 320.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "AND",
    },
    "AND2_X2": {
        "intrinsic_delay": 0.032,
        "input_cap": 2.10,
        "output_res": 160.0,
        "drive_strength": 2.0,
        "num_inputs": 2,
        "function": "AND",
    },
    "OR2_X1": {
        "intrinsic_delay": 0.042,
        "input_cap": 1.10,
        "output_res": 320.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "OR",
    },
    "OR2_X2": {
        "intrinsic_delay": 0.036,
        "input_cap": 2.20,
        "output_res": 160.0,
        "drive_strength": 2.0,
        "num_inputs": 2,
        "function": "OR",
    },
    "XOR2_X1": {
        "intrinsic_delay": 0.055,
        "input_cap": 1.50,
        "output_res": 400.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "XOR",
    },
    "XNOR2_X1": {
        "intrinsic_delay": 0.055,
        "input_cap": 1.50,
        "output_res": 400.0,
        "drive_strength": 1.0,
        "num_inputs": 2,
        "function": "XNOR",
    },
    "MUX2_X1": {
        "intrinsic_delay": 0.048,
        "input_cap": 1.30,
        "output_res": 360.0,
        "drive_strength": 1.0,
        "num_inputs": 3,
        "function": "MUX",
    },
    "AOI21_X1": {
        "intrinsic_delay": 0.030,
        "input_cap": 1.05,
        "output_res": 370.0,
        "drive_strength": 1.0,
        "num_inputs": 3,
        "function": "AOI",
    },
    "OAI21_X1": {
        "intrinsic_delay": 0.032,
        "input_cap": 1.10,
        "output_res": 380.0,
        "drive_strength": 1.0,
        "num_inputs": 3,
        "function": "OAI",
    },
}

# Wire model parameters (45nm technology)
WIRE_CAP_PER_UM = 0.00017  # fF/um -- unit wire capacitance
WIRE_RES_PER_UM = 0.12  # ohm/um -- unit wire resistance
DEFAULT_WIRE_LENGTH_UM = 10.0  # um -- default wire length estimate

# Gate type encoding (for one-hot features)
GATE_TYPES = sorted(set(c["function"] for c in NANGATE45_CELLS.values()))
GATE_TYPE_TO_IDX = {g: i for i, g in enumerate(GATE_TYPES)}
NUM_GATE_TYPES = len(GATE_TYPES)

# Cell names for random sampling
CELL_NAMES = list(NANGATE45_CELLS.keys())


def get_cell(name: str) -> dict:
    """Return cell parameters by name."""
    return NANGATE45_CELLS[name]


def compute_wire_delay(wire_length_um: float, load_cap_fF: float) -> float:
    """Elmore delay model for wire: tau = R_wire * (C_wire/2 + C_load)."""
    r_wire = WIRE_RES_PER_UM * wire_length_um
    c_wire = WIRE_CAP_PER_UM * wire_length_um
    return r_wire * (c_wire / 2.0 + load_cap_fF) * 1e-3  # convert to ns


def compute_gate_delay(cell_name: str, total_load_cap_fF: float) -> float:
    """Gate delay = intrinsic_delay + output_res * load_cap."""
    cell = NANGATE45_CELLS[cell_name]
    load_dependent = cell["output_res"] * total_load_cap_fF * 1e-6  # ohm * fF -> ns
    return cell["intrinsic_delay"] + load_dependent
