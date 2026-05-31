"""Trajectory builders and parameter-sampling utilities."""
from __future__ import annotations

import numpy as np
from itertools import combinations, permutations
from .core import ket0, kron_all, ry

# (psiaudit_core symbols already loaded)


def trajectory_A_sector_confined(apply_fn, n, L, n_params_fn, T=40, seed=0):
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    base = rng.uniform(-np.pi, np.pi, size=P)
    delta = rng.normal(0.0, 0.5, size=P)
    states = []
    for t in np.linspace(0, 2 * np.pi, T):
        theta = base + delta * np.sin(t) + 0.3 * np.cos(2 * t)
        states.append(apply_fn(theta, n, L))
    return np.array(states)


def trajectory_B_multi_sector(apply_fn, n, L, n_params_fn, T=40, seed=0,
                              init_spread=1.0):
    """Multi-sector trajectory builder, product-state initialisation.

    Matches Equation (7) of the v12 manuscript. At trajectory step t, the
    initial state is a product of independent single-qubit y-axis rotations
    applied to |0..0>:

        |psi_0(t)> = prod_q Ry( alpha_q(t) ) |0..0>
        alpha_q(t) = (pi/2) * ( 0.5 + 0.4 * sin( t + 2 pi q / n ) )

    The product state is not, in general, supported in a single SU(2)
    total-spin sector, so it activates more than one j when audited
    against SU(2). The typical per-qubit excitation probability is
    approximately 0.15, which places the expected U(1) Hamming-weight
    peak in the k = 1 sector for n = 6. See trajectory_B_dicke below for
    a symmetric Dicke alternative that lies in the j = n/2 sector.
    """
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    base = rng.uniform(-np.pi, np.pi, size=P)
    delta = rng.normal(0.0, 0.5, size=P)
    states = []
    for t in np.linspace(0, 2 * np.pi, T):
        theta = base + delta * np.sin(t) + 0.3 * np.cos(2 * t)
        alphas = init_spread * (np.pi / 2) * (
            0.5 + 0.4 * np.sin(t + 2 * np.pi * np.arange(n) / n)
        )
        U_init = kron_all([ry(a) for a in alphas])
        psi0 = U_init @ ket0(n)
        states.append(apply_fn(theta, n, L, init=psi0))
    return np.array(states)


def _symmetric_dicke_state(n, k):
    """Return the normalised symmetric Dicke state |W_k> on n qubits.

    |W_k> = (1/sqrt(C(n,k))) sum over all bit-strings of weight k.
    This state lies entirely in the j = n/2 total-spin sector.
    """
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    indices = []
    for positions in combinations(range(n), k):
        bitstring = 0
        for p in positions:
            bitstring |= (1 << (n - 1 - p))
        indices.append(bitstring)
    psi[indices] = 1.0
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        return psi
    return psi / norm


def trajectory_B_dicke(apply_fn, n, L, n_params_fn, T=40, seed=0):
    """Symmetric-Dicke alternative initialisation for Regime B.

    Initial state is a binomial superposition of symmetric Dicke states,

        |psi_0> = sum_{k=0}^{n} sqrt( C(n, k) / 2^n ) |W_k>,

    which lies entirely in the j = n/2 fully-symmetric SU(2) sector. Use
    this alternative when auditing claims about SU(2) sector preservation
    in isolation from product-state mixing artefacts.
    """
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    base = rng.uniform(-np.pi, np.pi, size=P)
    delta = rng.normal(0.0, 0.5, size=P)

    # Build fixed Dicke superposition once
    from math import comb
    dim = 2 ** n
    psi0 = np.zeros(dim, dtype=complex)
    for k in range(n + 1):
        psi0 += np.sqrt(comb(n, k) / (2 ** n)) * _symmetric_dicke_state(n, k)
    psi0 = psi0 / float(np.linalg.norm(psi0))

    states = []
    for t in np.linspace(0, 2 * np.pi, T):
        theta = base + delta * np.sin(t) + 0.3 * np.cos(2 * t)
        states.append(apply_fn(theta, n, L, init=psi0))
    return np.array(states)


def random_parameter_samples(apply_fn, n, L, n_params_fn, S=200, seed=1,
                             multi_sector=False):
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    states = []
    for _ in range(S):
        theta = rng.uniform(-np.pi, np.pi, size=P)
        if multi_sector:
            alphas = rng.uniform(0, np.pi, size=n)
            U_init = kron_all([ry(a) for a in alphas])
            psi0 = U_init @ ket0(n)
            states.append(apply_fn(theta, n, L, init=psi0))
        else:
            states.append(apply_fn(theta, n, L))
    return np.array(states)


def parameter_shift_gradient(apply_fn, n, L, theta, observable, param_idx,
                             shift=np.pi / 2, init=None):
    theta_p = theta.copy()
    theta_p[param_idx] += shift
    psi_p = apply_fn(theta_p, n, L, init=init)
    e_p = float(np.real(np.conj(psi_p) @ observable @ psi_p))

    theta_m = theta.copy()
    theta_m[param_idx] -= shift
    psi_m = apply_fn(theta_m, n, L, init=init)
    e_m = float(np.real(np.conj(psi_m) @ observable @ psi_m))
    return 0.5 * (e_p - e_m)


def gradient_variance_diagnostic(apply_fn, n, L, n_params_fn, observable,
                                 S=80, param_idx=0, seed=2,
                                 multi_sector=False, eps=1e-4):
    """Gradient variance via central finite difference.

    The standard parameter-shift rule with shift pi/2 only applies to gates
    whose generator has eigenvalues {-1, +1}. The Heisenberg pair generator
    used in the SU(2)-equivariant ansatz has eigenvalue spectrum
    {-3, +1, +1, +1}, so the pi/2 rule returns machine-zero rather than the
    true gradient. Finite difference at small eps applies uniformly across
    all generator types and is what the package uses.
    """
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    grads = []
    for _ in range(S):
        theta = rng.uniform(-np.pi, np.pi, size=P)
        if multi_sector:
            alphas = rng.uniform(0, np.pi, size=n)
            U_init = kron_all([ry(a) for a in alphas])
            psi0 = U_init @ ket0(n)
        else:
            psi0 = None
        e_vec = np.zeros(P)
        e_vec[param_idx] = 1.0
        psi_p = apply_fn(theta + eps * e_vec, n, L, init=psi0)
        psi_m = apply_fn(theta - eps * e_vec, n, L, init=psi0)
        g = (float(np.real(psi_p.conj() @ observable @ psi_p))
             - float(np.real(psi_m.conj() @ observable @ psi_m))) / (2 * eps)
        grads.append(g)
    return float(np.var(grads))
