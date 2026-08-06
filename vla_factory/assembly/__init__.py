"""Assembly layer: data × model × robot composition resolution.

Hosts the transform pipeline (moved from ``data/transforms``) and the
composition resolver (``resolver/``). The resolver combines the three
unified descriptions (DataSchema, ModelMetadata, RobotProfile) into a
``ResolvedAssembly``; downstream layers (training / inference) consume it.
"""
