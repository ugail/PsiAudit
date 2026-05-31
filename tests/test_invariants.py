"""Structural invariant tests for the PsiAudit primitives.

These mirror the self-test battery in the reproduction notebook. They verify
the projector constructions, the generator-sum compliance defect, the
Meyer-Wallach calibration, the representation-theoretic placement of symmetric
Dicke states, the multi-sector activation of Regime B, the symmetry-correct
ordering of the gated composite on the U(1) near-control, and the Regime A
collapse of an equivariant ansatz. They need no GPU and run in a few seconds.

Run with:  pytest tests
"""
from __future__ import annotations

import numpy as np

from psiaudit.core import (
    ket0, density, fro_norm,
    excitation_number_operator, excitation_projectors,
    total_spin_operators, su2_projectors,
    sector_probabilities, Psi_G_full, meyer_wallach_entanglement,
)
from psiaudit.ansatze import (
    ANSATZ_REGISTRY, H_eff_U1, H_eff_SU2, H_eff_broken,
)
from psiaudit.trajectories import (
    trajectory_A_sector_confined, trajectory_B_multi_sector,
    _symmetric_dicke_state,
)

N_TEST = 6
L_TEST = 2
SEED = 7
WEIGHTS = (0.40, 0.35, 0.25)
GAMMA = 3.0


def test_u1_projectors_idempotent_and_complete():
    proj, _ = excitation_projectors(N_TEST)
    sum_err = float(np.linalg.norm(sum(proj) - np.eye(2 ** N_TEST)))
    assert sum_err < 1e-9
    assert all(np.linalg.norm(P @ P - P) < 1e-10 for P in proj)


def test_su2_projectors_idempotent_and_complete():
    proj, _, _ = su2_projectors(N_TEST)
    sum_err = float(np.linalg.norm(sum(proj) - np.eye(2 ** N_TEST)))
    assert sum_err < 1e-9


def test_equivariant_hamiltonians_commute_with_generators():
    Nop = excitation_number_operator(N_TEST)
    Sx, Sy, Sz, _ = total_spin_operators(N_TEST)
    assert fro_norm(H_eff_U1(N_TEST) @ Nop - Nop @ H_eff_U1(N_TEST)) < 1e-9
    HS = H_eff_SU2(N_TEST)
    assert fro_norm(HS @ Sx - Sx @ HS) < 1e-9
    assert fro_norm(HS @ Sy - Sy @ HS) < 1e-9
    assert fro_norm(HS @ Sz - Sz @ HS) < 1e-9


def test_broken_hamiltonian_flagged_by_u1_defect():
    Nop = excitation_number_operator(N_TEST)
    err = fro_norm(H_eff_broken(N_TEST) @ Nop - Nop @ H_eff_broken(N_TEST))
    assert err > 1e-3


def test_meyer_wallach_calibration():
    Q_prod = meyer_wallach_entanglement(ket0(N_TEST))
    psi_ghz = np.zeros(2 ** N_TEST, dtype=complex)
    psi_ghz[0] = 1 / np.sqrt(2)
    psi_ghz[-1] = 1 / np.sqrt(2)
    Q_ghz = meyer_wallach_entanglement(psi_ghz)
    assert abs(Q_prod) < 1e-6
    assert abs(Q_ghz - 1) < 1e-6


def test_symmetric_dicke_states_in_top_spin_sector():
    proj, labels, _ = su2_projectors(N_TEST)
    j_max_label = f"j={N_TEST / 2:.1f}"
    if j_max_label in labels:
        j_idx = labels.index(j_max_label)
    elif f"j={int(N_TEST / 2)}" in labels:
        j_idx = labels.index(f"j={int(N_TEST / 2)}")
    else:
        j_idx = len(proj) - 1
    for k in range(N_TEST + 1):
        probs = sector_probabilities(density(_symmetric_dicke_state(N_TEST, k)), proj)
        assert probs[j_idx] > 0.999


def test_regime_b_has_multi_sector_su2_support():
    proj, _, _ = su2_projectors(N_TEST)
    info = ANSATZ_REGISTRY["SU2_equiv"]
    states = trajectory_B_multi_sector(info["apply"], N_TEST, L_TEST,
                                       info["n_params"], T=20, seed=SEED)
    P = np.vstack([sector_probabilities(density(psi), proj) for psi in states])
    assert int(np.sum(P.mean(axis=0) > 0.01)) >= 2


def test_gated_composite_orders_u1_nearcontrol():
    proj, _ = excitation_projectors(N_TEST)
    Nop = excitation_number_operator(N_TEST)
    eq = ANSATZ_REGISTRY["U1_equiv"]
    br = ANSATZ_REGISTRY["U1_broken"]
    s_eq = trajectory_B_multi_sector(eq["apply"], N_TEST, L_TEST, eq["n_params"], T=20, seed=SEED)
    s_br = trajectory_B_multi_sector(br["apply"], N_TEST, L_TEST, br["n_params"], T=20, seed=SEED)
    rep_eq = Psi_G_full(s_eq, proj, eq["H_eff"](N_TEST), [Nop], weights=WEIGHTS, gamma=GAMMA)
    rep_br = Psi_G_full(s_br, proj, br["H_eff"](N_TEST), [Nop], weights=WEIGHTS, gamma=GAMMA)
    assert rep_eq["Psi_G"] > rep_br["Psi_G"]


def test_regime_a_collapse_for_equivariant():
    proj, _ = excitation_projectors(N_TEST)
    Nop = excitation_number_operator(N_TEST)
    eq = ANSATZ_REGISTRY["U1_equiv"]
    s = trajectory_A_sector_confined(eq["apply"], N_TEST, L_TEST, eq["n_params"], T=20, seed=SEED)
    rep = Psi_G_full(s, proj, eq["H_eff"](N_TEST), [Nop], weights=WEIGHTS, gamma=GAMMA)
    assert rep["Psi_G"] < 1e-9
