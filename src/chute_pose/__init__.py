"""Geometry-based pose prediction for the BiBaZu chute."""

from .frame import ChuteFrame
from .geometry import GeometryReport, GeometryValidationError, inspect_mesh, load_solid_mesh
from .contacts import ContactPose, PoseCatalog, SupportFace, build_pose_catalog
from .stability import (
    FrictionEstimate,
    PoseStability,
    StabilityAnalysis,
    StabilitySample,
    analyze_pose_stability,
    estimate_equal_contact_friction,
)

__all__ = [
    "ChuteFrame",
    "GeometryReport",
    "GeometryValidationError",
    "inspect_mesh",
    "load_solid_mesh",
    "ContactPose",
    "PoseCatalog",
    "SupportFace",
    "build_pose_catalog",
    "FrictionEstimate",
    "PoseStability",
    "StabilityAnalysis",
    "StabilitySample",
    "analyze_pose_stability",
    "estimate_equal_contact_friction",
]
