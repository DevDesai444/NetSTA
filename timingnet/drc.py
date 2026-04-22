"""
Simplified DRC-hotspot labelling.

Each grid cell (GCell analog) is modelled as having a fixed routing-track
capacity. Nodes whose containing cell carries RUDY demand above
`hotspot_factor * capacity` are flagged as DRC-violation candidates.

For Nangate45 we approximate with 10 tracks per GCell per metal layer; this
is a simplification but produces realistic class-imbalance for the
DRC-head focal loss to learn against.
"""

from typing import Dict, List, Optional

from .circuit_gen import Circuit
from .congestion import compute_demand_grid


# Nangate45 routing tracks per GCell per metal layer (approximation).
TRACKS_PER_CELL = 10
# Demand fraction at which we flag a hotspot.
HOTSPOT_FACTOR = 0.9


def compute_drc_labels(
    circuit: Circuit,
    demand_grid: Optional[List[List[float]]] = None,
    tracks_per_cell: int = TRACKS_PER_CELL,
    hotspot_factor: float = HOTSPOT_FACTOR,
) -> Dict[str, float]:
    """Per-node binary DRC-hotspot labels.

    Pass an existing demand grid (e.g. from compute_rudy_congestion) to
    avoid recomputing demand. Returns a {node_id: 0.0/1.0} dict.
    """
    if demand_grid is None:
        demand_grid, _ = compute_demand_grid(circuit)
    threshold = tracks_per_cell * hotspot_factor

    labels: Dict[str, float] = {}
    grid_w = len(demand_grid)
    grid_h = len(demand_grid[0]) if grid_w > 0 else 0
    for nid in circuit.nodes:
        pos = circuit.positions.get(nid)
        if pos is None:
            labels[nid] = 0.0
            continue
        x, y = pos
        if 0 <= x < grid_w and 0 <= y < grid_h:
            labels[nid] = 1.0 if demand_grid[x][y] > threshold else 0.0
        else:
            labels[nid] = 0.0
    return labels
