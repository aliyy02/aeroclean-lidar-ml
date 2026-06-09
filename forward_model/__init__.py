"""First-principles LiDAR forward model.

The simulator (Standard Lidar / SensorType 6, ComputerVision mode) gives geometry only
-- where each beam lands + the actor name it hit. This package adds the sensor's optics
per point in pure Python: distance, incidence angle, per-material intensity, and a
return / no-return decision, calibrated once against real test-bed scans.

Public API:
    from forward_model import apply_forward_model, load_constants, default_constants
    res = apply_forward_model(xyz, names, normals, load_constants("constants_calibrated.yaml"))
    # res.xyz (K,3), res.intensity (K,), res.labels (K,)

See lidar_post_processing.pdf for the physics and forward_model/README.md for usage.
"""
from .apply import apply_forward_model, ScanResult
from .constants import default_constants, load_constants, ForwardModelConstants, MaterialParams
from . import materials

__all__ = [
    "apply_forward_model",
    "ScanResult",
    "default_constants",
    "load_constants",
    "ForwardModelConstants",
    "MaterialParams",
    "materials",
]
