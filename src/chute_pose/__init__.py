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
from .symmetry import (
    PoseEquivalenceClass,
    RotationalSymmetryGroup,
    SymmetryElement,
    SymmetryReducedCatalog,
    build_symmetry_reduced_catalog,
    detect_rotational_symmetry,
    reduce_catalog_by_symmetry,
)
from .step_verification import (
    StepSupportUnavailable,
    StepSymmetryElementCheck,
    StepSymmetryVerification,
    verify_step_symmetry,
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
    "PoseEquivalenceClass",
    "RotationalSymmetryGroup",
    "SymmetryElement",
    "SymmetryReducedCatalog",
    "build_symmetry_reduced_catalog",
    "detect_rotational_symmetry",
    "reduce_catalog_by_symmetry",
    "StepSupportUnavailable",
    "StepSymmetryElementCheck",
    "StepSymmetryVerification",
    "verify_step_symmetry",
]
