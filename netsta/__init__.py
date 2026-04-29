"""
netsta — shim package exposing the timingnet RAG + model API under the
project-facing name. All real implementation lives in `timingnet/`.
"""

from timingnet import *  # noqa: F401,F403
from timingnet.rag import (  # noqa: F401
    CircuitSpec,
    DeviceSpec,
    DesignReport,
    Bottleneck,
    KnowledgeStore,
    advise,
    generate_from_spec,
    parse_to_spec,
    run_prediction,
)
