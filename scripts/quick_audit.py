#!/usr/bin/env python3
"""Minimal worked example: audit a user-supplied ansatz.

This is the shortest path from a parameterised circuit to the four PsiAudit
components and the composite. It audits one of the built-in reference ansaetze
against its natural target group, then shows how to drop in your own circuit.

Run with:
    python scripts/quick_audit.py
"""
from __future__ import annotations

import numpy as np

import psiaudit as pa


def audit_builtin(name="Sn_equiv", n=6, L=2, T=40, seed=7):
    """Audit a built-in reference ansatz against its natural target group."""
    info = pa.ANSATZ_REGISTRY[name]
    group_name = pa.NATURAL_GROUP[name]
    groups = pa.build_groups_at(n)
    g = groups[group_name]

    result = pa.audit_one(
        name, info, n, L, group_name,
        g["projectors"], g["generators"], T,
        pa.trajectory_B_multi_sector,
        gamma=3.0, weights=(0.40, 0.35, 0.25), seed=seed,
    )

    print(f"Audit of {name} against {group_name} (Regime B, n={n}, L={L}, T={T})")
    print("-" * 56)
    for k in ["H_G", "D_G", "M_G", "S_G", "Psi_G"]:
        print(f"  {k:6s} = {result[k]:.4f}")
    return result


def audit_custom():
    """Audit a custom circuit.

    A circuit is any callable ``apply(params, n, L)`` that returns the output
    state vector of length ``2**n``. You also supply the number of parameters
    via ``n_params(n, L)`` and the effective generator-sum Hamiltonian via
    ``H_eff(n)``. Below we reuse the built-in U(1)-equivariant primitives to
    show the registry-entry shape you need to provide.
    """
    from psiaudit.ansatze import apply_U1, n_params_U1, H_eff_U1

    custom = {
        "apply": apply_U1,        # your circuit: (params, n, L) -> state vector
        "n_params": n_params_U1,  # (n, L) -> int
        "H_eff": H_eff_U1,        # (n) -> generator-sum Hamiltonian matrix
    }

    n, L, T = 6, 2, 40
    groups = pa.build_groups_at(n)
    g = groups["U(1)"]            # choose the target group to audit against
    result = pa.audit_one(
        "custom", custom, n, L, "U(1)",
        g["projectors"], g["generators"], T,
        pa.trajectory_B_multi_sector,
        gamma=3.0, weights=(0.40, 0.35, 0.25), seed=0,
    )
    print("\nAudit of a custom (here U(1)-equivariant) circuit against U(1)")
    print("-" * 56)
    for k in ["H_G", "D_G", "M_G", "S_G", "Psi_G"]:
        print(f"  {k:6s} = {result[k]:.4f}")
    return result


if __name__ == "__main__":
    audit_builtin()
    audit_custom()
