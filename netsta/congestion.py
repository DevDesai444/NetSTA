"""
RUDY (Rectangular Uniform wire DensitY) routing-congestion estimator.

For each net, distribute its routing demand uniformly across the pin
bounding box; per-cell demand is the sum of per-net contributions covering
that cell. Per-node congestion is the demand at the cell where the node is
placed, normalized to [0, 1].

Reference:
  Spindler, Schlichtmann. "Fast and Accurate Routing Demand Estimation for
  Efficient Routability-driven Placement." DATE 2007.
"""

from typing import Dict, List, Tuple

from .circuit_gen import Circuit


def compute_demand_grid(circuit: Circuit) -> Tuple[List[List[float]], Tuple[int, int]]:
    """Return (demand_grid, (grid_w, grid_h)).

    demand_grid is a 2D list indexed as demand[x][y] giving the summed RUDY
    routing demand at each grid cell.
    """
    positions = circuit.positions
    grid_w, grid_h = circuit.grid_size
    if grid_w <= 0 or grid_h <= 0 or not positions:
        return [[0.0]], (1, 1)

    demand = [[0.0] * grid_h for _ in range(grid_w)]

    for net in circuit.nets.values():
        pins = [net.driver] + list(net.sinks)
        pin_xy = [positions[p] for p in pins if p in positions]
        if len(pin_xy) < 2:
            continue
        xs = [p[0] for p in pin_xy]
        ys = [p[1] for p in pin_xy]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        hpwl = (xmax - xmin) + (ymax - ymin)
        if hpwl == 0:
            continue
        fanout = max(1, len(net.sinks))
        bbox_w = xmax - xmin + 1
        bbox_h = ymax - ymin + 1
        # Per-cell intensity: total demand (HPWL * fanout) distributed
        # uniformly over the bounding-box area.
        per_cell = (hpwl * fanout) / (bbox_w * bbox_h)
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                demand[x][y] += per_cell

    return demand, (grid_w, grid_h)


def compute_rudy_congestion(circuit: Circuit) -> Tuple[Dict[str, float], List[List[float]]]:
    """Return (per_node_normalized_congestion, demand_grid).

    The grid is returned so callers (e.g. drc.py) can apply absolute
    capacity thresholds without re-deriving demand.
    """
    demand, (grid_w, grid_h) = compute_demand_grid(circuit)
    raw: Dict[str, float] = {}
    for nid in circuit.nodes:
        pos = circuit.positions.get(nid)
        if pos is None:
            raw[nid] = 0.0
            continue
        x, y = pos
        if 0 <= x < grid_w and 0 <= y < grid_h:
            raw[nid] = demand[x][y]
        else:
            raw[nid] = 0.0
    max_d = max(raw.values()) if raw else 1.0
    max_d = max(max_d, 1e-6)
    normalized = {nid: v / max_d for nid, v in raw.items()}
    return normalized, demand
