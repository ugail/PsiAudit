"""Reference ansatz library and effective generator-sum Hamiltonians."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, eigh
from itertools import combinations, permutations
from .core import X, Y, Z, ket0, normalize, op_on_qubit, rx, ry, rz, two_qubit_op

# (psiaudit_core symbols already loaded in the previous cell)


# ---------- helpers ----------

def n_layers_from_p(P, per_layer):
    return P // per_layer


# ---------- 1. Hardware-efficient ansatz (HEA) ----------

def n_params_HEA(n, L):
    return 3 * n * L



# ---------------------------------------------------------------------------
# Fast state-vector gate application.
# Gates are applied by reshaping the state into an n-axis tensor and
# contracting only the affected qubit axes, instead of constructing dense
# 2**n x 2**n operators. This is mathematically identical to the dense
# construction (validated to machine precision) and is orders of magnitude
# faster at n >= 8. Two-qubit generators used here (XY-hop, ZZ, Heisenberg)
# are symmetric under qubit exchange, so edge ordering is immaterial.
# ---------------------------------------------------------------------------

def _apply_1q(psi, U, q, n):
    """Apply a single-qubit gate U to qubit q of an n-qubit state vector."""
    t = psi.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1)


def _apply_2q(psi, U4, q1, q2, n):
    """Apply a two-qubit 4x4 gate U4 (acting on (q1, q2)) to a state vector."""
    t = psi.reshape([2] * n)
    U = U4.reshape(2, 2, 2, 2)            # out1, out2, in1, in2
    t = np.tensordot(U, t, axes=([2, 3], [q1, q2]))
    t = np.moveaxis(t, [0, 1], [q1, q2])
    return t.reshape(-1)


# Two-qubit generator matrices (4x4), exponentiated once per angle.
_G_HOP_XY = 0.5 * (np.kron(X, X) + np.kron(Y, Y))
_G_ZZ = np.kron(Z, Z)
_G_HEIS = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
_CZ4 = np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=complex))

def _u_hop_xy(theta):
    return expm(-1j * theta * _G_HOP_XY)

def _u_zz(theta):
    return expm(-1j * theta * _G_ZZ)

def _u_heis(theta):
    return expm(-1j * theta * _G_HEIS)


def apply_HEA(theta, n, L, init=None):
    """RX-RY-RZ on each qubit followed by CZ ring, repeated L times.
    theta has length 3*n*L. Optional init overrides the |0...0> default."""
    psi = ket0(n) if init is None else normalize(np.asarray(init, dtype=complex))
    idx = 0
    for layer in range(L):
        for q in range(n):
            ax, ay, az = theta[idx], theta[idx + 1], theta[idx + 2]
            idx += 3
            psi = _apply_1q(psi, rz(az), q, n)
            psi = _apply_1q(psi, ry(ay), q, n)
            psi = _apply_1q(psi, rx(ax), q, n)
        for q in range(n - 1):
            psi = _apply_2q(psi, _CZ4, q, q + 1, n)
        if n > 2:
            psi = _apply_2q(psi, _CZ4, n - 1, 0, n)
    return normalize(psi)

# ---------- 2. U(1)-equivariant ansatz (XY-conserving) ----------

def hop_xy_generator(i, j, n):
    """Hopping generator (X_i X_j + Y_i Y_j)/2 - preserves total Z (U(1))."""
    return 0.5 * (two_qubit_op(X, i, X, j, n) + two_qubit_op(Y, i, Y, j, n))


def zz_generator(i, j, n):
    """Z_i Z_j - also U(1)-preserving."""
    return two_qubit_op(Z, i, Z, j, n)


def n_params_U1(n, L):
    # Per layer: n single-qubit Z rotations + (n-1)+1 hopping + (n-1)+1 ZZ
    edges = n if n > 2 else n - 1
    return L * (n + 2 * edges)


def apply_U1(theta, n, L, init=None):
    """U(1)-equivariant: Z rotations + nearest-neighbour XY hop and ZZ."""
    psi = ket0(n) if init is None else normalize(np.asarray(init, dtype=complex))
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    idx = 0
    for layer in range(L):
        for q in range(n):
            psi = _apply_1q(psi, rz(theta[idx]), q, n)
            idx += 1
        for (i, j) in edges:
            psi = _apply_2q(psi, _u_hop_xy(theta[idx]), i, j, n)
            idx += 1
        for (i, j) in edges:
            psi = _apply_2q(psi, _u_zz(theta[idx]), i, j, n)
            idx += 1
    return normalize(psi)

# ---------- 3. SU(2)-equivariant ansatz (Heisenberg-block) ----------

def heisenberg_pair_generator(i, j, n):
    """SU(2)-invariant pair generator X_i X_j + Y_i Y_j + Z_i Z_j."""
    return (two_qubit_op(X, i, X, j, n)
            + two_qubit_op(Y, i, Y, j, n)
            + two_qubit_op(Z, i, Z, j, n))


def n_params_SU2(n, L):
    edges = n if n > 2 else n - 1
    return L * edges


def apply_SU2(theta, n, L, init=None):
    """SU(2)-equivariant: only Heisenberg-pair generators on edges (no
    single-qubit rotations, since those break SU(2))."""
    psi = ket0(n) if init is None else normalize(np.asarray(init, dtype=complex))
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    idx = 0
    for layer in range(L):
        for (i, j) in edges:
            psi = _apply_2q(psi, _u_heis(theta[idx]), i, j, n)
            idx += 1
    return normalize(psi)

# ---------- 4. S_n permutation-equivariant ansatz (parameter-input) ----------

def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def n_params_Sn(n, L):
    # Per layer: 1 parameter shared across all single-qubit Y rotations
    # (must be the same to preserve S_n) + 1 parameter shared across all
    # ZZ pairs. So 2 parameters per layer.
    return 2 * L


def apply_Sn(theta, n, L, init=None):
    """S_n-equivariant: parameter-input form. Per layer apply RY(alpha) on
    every qubit (shared angle) followed by RZZ(beta) on every pair (shared
    angle). Both operations commute with all permutations."""
    psi = ket0(n) if init is None else normalize(np.asarray(init, dtype=complex))
    pairs = all_pairs(n)
    idx = 0
    for layer in range(L):
        alpha = theta[idx]
        idx += 1
        for q in range(n):
            psi = _apply_1q(psi, ry(alpha), q, n)
        beta = theta[idx]
        idx += 1
        for (i, j) in pairs:
            psi = _apply_2q(psi, _u_zz(beta), i, j, n)
    return normalize(psi)

# ---------- 5. Symmetry-broken ansatz (perturbed equivariant) ----------

def n_params_broken(n, L):
    return n_params_U1(n, L)


def apply_broken(theta, n, L, breaking_strength=0.4, init=None):
    """Take the U(1) ansatz parameters but inject local-X rotations at the
    end of each layer using a fixed (non-trainable) angle scaled by the
    breaking_strength. This deliberately violates U(1) at the channel level."""
    psi = ket0(n) if init is None else normalize(np.asarray(init, dtype=complex))
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    idx = 0
    for layer in range(L):
        for q in range(n):
            psi = _apply_1q(psi, rz(theta[idx]), q, n)
            idx += 1
        for (i, j) in edges:
            psi = _apply_2q(psi, _u_hop_xy(theta[idx]), i, j, n)
            idx += 1
        for (i, j) in edges:
            psi = _apply_2q(psi, _u_zz(theta[idx]), i, j, n)
            idx += 1
        for q in range(n):
            psi = _apply_1q(psi, rx(breaking_strength), q, n)
    return normalize(psi)

# ---------- Effective Hamiltonian per ansatz ----------
# For S_G compliance we need a single Hermitian operator that "represents"
# the ansatz's circuit-level structure. We use the sum of generators of all
# rotation gates the ansatz uses, which captures whether any one of them
# breaks the symmetry.

def H_eff_HEA(n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n):
        H += op_on_qubit(X, q, n) + op_on_qubit(Y, q, n) + op_on_qubit(Z, q, n)
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    return H


def H_eff_U1(n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n):
        H += op_on_qubit(Z, q, n)
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    for (i, j) in edges:
        H += hop_xy_generator(i, j, n)
        H += zz_generator(i, j, n)
    return H


def H_eff_SU2(n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    edges = [(i, i + 1) for i in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    for (i, j) in edges:
        H += heisenberg_pair_generator(i, j, n)
    return H


def H_eff_Sn(n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n):
        H += op_on_qubit(Y, q, n)
    for (i, j) in all_pairs(n):
        H += two_qubit_op(Z, i, Z, j, n)
    return H


def H_eff_broken(n, breaking_strength=0.4):
    """Effective Hamiltonian for the broken ansatz: same as U(1) plus a
    local-X piece scaled by the breaking strength."""
    H = H_eff_U1(n)
    for q in range(n):
        H += breaking_strength * op_on_qubit(X, q, n)
    return H


# ---------- Registry ----------

ANSATZ_REGISTRY = {
    "HEA": {
        "apply": apply_HEA,
        "n_params": n_params_HEA,
        "H_eff": H_eff_HEA,
        "natural_group": None,
        "description": "Hardware-efficient ansatz (RX-RY-RZ + CZ ring), non-equivariant baseline",
    },
    "U1_equiv": {
        "apply": apply_U1,
        "n_params": n_params_U1,
        "H_eff": H_eff_U1,
        "natural_group": "U(1)",
        "description": "U(1)-equivariant ansatz preserving total excitation number",
    },
    "SU2_equiv": {
        "apply": apply_SU2,
        "n_params": n_params_SU2,
        "H_eff": H_eff_SU2,
        "natural_group": "SU(2)",
        "description": "SU(2)-equivariant ansatz preserving total spin",
    },
    "Sn_equiv": {
        "apply": apply_Sn,
        "n_params": n_params_Sn,
        "H_eff": H_eff_Sn,
        "natural_group": "S_n",
        "description": "S_n-permutation-equivariant ansatz (parameter-input, shared angles)",
    },
    "U1_broken": {
        "apply": apply_broken,
        "n_params": n_params_broken,
        "H_eff": H_eff_broken,
        "natural_group": "U(1)",
        "description": "U(1) ansatz with local-X breaking suffix (negative control)",
    },
}
