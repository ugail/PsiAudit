"""PsiAudit: a toolkit for auditing symmetry-organised complexity in
equivariant quantum neural network ansaetze.

The package exposes four normalised audit components (sector-occupation
entropy ``H_G``, cross-sector coherence ``D_G``, sectoral fluctuation ``M_G``,
and generator-sum compliance ``S_G``), their configurable composite ``Psi_G``,
a reference ansatz library, two trajectory builders, and the standard
comparative diagnostics, together with the single-audit driver.

See the README and the reproduction notebook for worked examples. The
mathematical primitives are documented in the companion paper,
Ugail and Howard, Symmetry 2026, 18(6), 912.
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import core, ansatze, trajectories, audit

# Audit components, composite, and diagnostics
from .core import (
    H_G_score,
    D_G_score,
    M_G_score,
    S_G_compliance,
    Psi_G_full,
    unitary_commutator_deviation,
    expressibility_kl,
    gradient_variance,
    meyer_wallach_entanglement,
    excitation_projectors,
    su2_projectors,
    sn_orbit_projectors,
)

# Reference ansatz library
from .ansatze import ANSATZ_REGISTRY

# Trajectory builders
from .trajectories import (
    trajectory_A_sector_confined,
    trajectory_B_multi_sector,
)

# Audit driver and group construction
from .audit import audit_one, build_groups_at, NATURAL_GROUP

__all__ = [
    "H_G_score", "D_G_score", "M_G_score", "S_G_compliance",
    "Psi_G_full", "unitary_commutator_deviation",
    "expressibility_kl", "gradient_variance", "meyer_wallach_entanglement",
    "excitation_projectors", "su2_projectors", "sn_orbit_projectors",
    "ANSATZ_REGISTRY",
    "trajectory_A_sector_confined", "trajectory_B_multi_sector",
    "audit_one", "build_groups_at", "NATURAL_GROUP",
    "core", "ansatze", "trajectories", "audit",
]
