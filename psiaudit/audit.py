"""Group-projector construction and the single-audit driver.

These are the reusable building blocks that the experiment scripts call. The
heavy experimental loops live in the scripts and the reproduction notebook, not
here, so this module stays free of run configuration and global state.
"""
from __future__ import annotations

import numpy as np

from .core import (
    Psi_G_full,
    excitation_number_operator,
    excitation_projectors,
    su2_projectors,
    sn_orbit_projectors,
)

# Natural target group of each reference ansatz. The hardware-efficient
# baseline carries no symmetry constraint and is audited against U(1) only
# for the purpose of a like-for-like reference comparison.
NATURAL_GROUP = {
    "HEA":       "U(1)",
    "U1_equiv":  "U(1)",
    "SU2_equiv": "SU(2)",
    "Sn_equiv":  "S_n",
    "U1_broken": "U(1)",
}


def build_groups_at(n):
    """Build the three target-group sector decompositions at ``n`` qubits.

    Returns a dict keyed by group name. Each entry holds the sector
    ``projectors``, their ``labels``, and the ``generators`` used by the
    generator-sum compliance defect.
    """
    proj_u1, lab_u1 = excitation_projectors(n)
    Nop = excitation_number_operator(n)
    proj_su2, lab_su2, gens_su2 = su2_projectors(n)
    Sx_, Sy_, Sz_, _ = gens_su2
    proj_sn, lab_sn, gens_sn = sn_orbit_projectors(n)
    return {
        "U(1)":  dict(projectors=proj_u1,  labels=lab_u1,  generators=[Nop]),
        "SU(2)": dict(projectors=proj_su2, labels=lab_su2, generators=[Sx_, Sy_, Sz_]),
        "S_n":   dict(projectors=proj_sn,  labels=lab_sn,  generators=gens_sn),
    }


def audit_one(name, info, n, L, group_name, projectors, generators, T,
              traj_fn, gamma, weights, seed=0):
    """Audit one ansatz against one target group under one trajectory regime.

    ``info`` is an entry of :data:`psiaudit.ansatze.ANSATZ_REGISTRY`. The
    return value is a flat dict of the four components, the compliance defect,
    the raw and gated composite, and the run configuration, suitable for direct
    insertion into a :class:`pandas.DataFrame`.
    """
    states = traj_fn(info["apply"], n, L, info["n_params"], T=T, seed=seed)
    H_eff = info["H_eff"](n)
    rep = Psi_G_full(states, projectors, H_eff, generators,
                     weights=weights, gamma=gamma)
    return {
        "ansatz": name, "group": group_name, "regime": traj_fn.__name__,
        "n": n, "L": L, "T": T,
        "n_params": int(info["n_params"](n, L)),
        **{k: rep[k] for k in ["H_G", "D_G", "D_inter", "D_mult",
                                "M_G", "S_G", "defect", "Psi_raw", "Psi_G"]},
    }
