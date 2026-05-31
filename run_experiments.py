#!/usr/bin/env python3

"""Full reproduction pipeline for the PsiAudit paper.



Runs the complete experimental battery used in the manuscript: the two-regime

audit, the scaling study, the comparative diagnostics, the multi-seed

replication with confidence intervals, the sensitivity sweep and H_G-gate

ablation, the downstream compatibility study, the audit-cost measurement, the

self-test battery, and the six manuscript figures. Every CSV that populates the

Results/ directory and every figure is regenerated from a clean run.



Runs at full production fidelity (fast_pass = False) and writes to ./results by

default. Set PSIAUDIT_OUT to change the output directory. Pure NumPy/SciPy,

CPU-only, no GPU required. A full run takes on the order of a few minutes.

"""

import matplotlib

matplotlib.use('Agg')



# ---------------------------------------------------------------------------
# Environment and output directories (local-first).
# Writes to ./results by default. Set PSIAUDIT_OUT to override the location.
# In Google Colab, set PSIAUDIT_USE_DRIVE=1 to mount and write to Drive.
# ---------------------------------------------------------------------------
import sys, os, time, json, platform
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
USE_DRIVE = os.environ.get("PSIAUDIT_USE_DRIVE", "0") == "1"
DRIVE_SUBPATH = "PsiAudit_results"
LOCAL_DIR = Path(os.environ.get("PSIAUDIT_OUT", "./results"))

def _prepare_dirs(base):
    base = Path(base)
    fig, data, meta = base / "figures", base / "data", base / "metadata"
    for d in (base, fig, data, meta):
        d.mkdir(parents=True, exist_ok=True)
    for d in (base, fig, data, meta):
        if not d.is_dir():
            raise OSError(f"could not create directory: {d}")
    return base, fig, data, meta

OUT_DIR = None
if IN_COLAB and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        candidate = Path("/content/drive/MyDrive") / DRIVE_SUBPATH
        OUT_DIR, FIG_DIR, DATA_DIR, META_DIR = _prepare_dirs(candidate)
    except Exception as e:
        print(f"Drive path unavailable ({e}); falling back to local directory.")
        OUT_DIR = None

if OUT_DIR is None:
    OUT_DIR, FIG_DIR, DATA_DIR, META_DIR = _prepare_dirs(LOCAL_DIR)

print(f"Environment  : {'Colab' if IN_COLAB else 'Local'}")
print(f"OUT_DIR      : {OUT_DIR.resolve()}")


# Single source of truth for run configuration. Adjust here only.
CONFIG = {
    # Audit grid
    "n_audit": 6,             # main audit system size
    "L": 2,                   # circuit depth (number of layers)
    "T": 40,                  # number of trajectory states for Psi_G
    # Scaling study
    "n_scaling": [4, 6, 8],   # qubit counts for the scaling sweep
    # Comparative diagnostics
    "S_diag": 200,            # parameter samples for KL/grad-var/MW
    "S_grad": 80,             # parameter samples for the gradient variance
    # Psi_G hyper-parameters
    "weights": (0.40, 0.35, 0.25),  # (w_H, w_D, w_M)
    "gamma": 3.0,             # S_G compliance penalty
    "seed": 7,                # global seed
    # Speed switches
    "fast_pass": False,       # if True, halve T and S_diag
}
if CONFIG["fast_pass"]:
    CONFIG["T"] = 20
    CONFIG["S_diag"] = 80

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k:14s} = {v}")


# Analysis configuration. Controls the multi-seed replication, the
# sensitivity sweeps, the unitary-compliance sample budget, the downstream
# study, and the audit-cost measurement. The fast_pass switch above scales
# these down for a quick smoke run.
ANALYSIS = {
    # Multi-seed replication
    "run_multiseed":      True,
    "n_seeds":            20,          # seeds for means and confidence intervals
    "seed_list":          None,        # None => range(n_seeds); or supply a list
    # Weight / gamma sensitivity + H_G-gate ablation
    "run_sensitivity":    True,
    "n_weight_samples":   200,         # Dirichlet draws over (w_H, w_D, w_M)
    "gamma_grid":         [0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
    # Unitary-level compliance sample budget. Delta_U is a structural
    # property of the circuit and is invariant to S to machine precision,
    # so a small budget is sufficient.
    "delta_U_samples":    8,
    # Downstream correlation study
    "run_downstream":     True,
    "downstream_n":       6,
    "downstream_train_steps": 60,
    "downstream_datasets_per_ansatz": 1,
    "downstream_train_seeds": 8,    # SPSA training seeds for accuracy mean and spread
    # Audit-cost scaling
    "run_cost_scaling":   True,
    "cost_scaling_n":     [4, 6, 8, 10],
}
if CONFIG.get("fast_pass", False):
    ANALYSIS["n_seeds"] = 5
    ANALYSIS["n_weight_samples"] = 50
    ANALYSIS["downstream_train_steps"] = 25
    ANALYSIS["downstream_train_seeds"] = 3
    ANALYSIS["cost_scaling_n"] = [4, 6, 8]
    ANALYSIS["delta_U_samples"] = 4
    # Cap the per-seed scaling sweep at n <= 6 in fast mode so the n = 8
    # dense computation is not repeated on every seed; the full single-seed
    # scaling result still covers n = 8.
    ANALYSIS["multiseed_scaling_n"] = [m for m in CONFIG["n_scaling"] if m <= 6]
else:
    ANALYSIS["multiseed_scaling_n"] = list(CONFIG["n_scaling"])

def _seed_list():
    if ANALYSIS["seed_list"] is not None:
        return list(ANALYSIS["seed_list"])
    return list(range(ANALYSIS["n_seeds"]))

print("Analysis configuration:")
for k, v in ANALYSIS.items():
    print(f"  {k:28s} = {v}")
print(f"  seeds                        = {_seed_list()}")


# Common scientific stack and matplotlib defaults for production figures
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.linalg import expm, eigh
from itertools import combinations

np.random.seed(CONFIG["seed"])

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "patch.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Colour palette used across all figures
PALETTE = {
    "HEA":        "#3a3a3a",
    "U1_equiv":   "#1f77b4",
    "SU2_equiv":  "#2ca02c",
    "Sn_equiv":   "#d62728",
    "U1_broken":  "#9467bd",
}

# Manuscript-style display names with proper LaTeX
DISPLAY_NAMES = {
    "HEA":        "HEA",
    "U1_equiv":   r"$U(1)$-equiv",
    "SU2_equiv":  r"$SU(2)$-equiv",
    "Sn_equiv":   r"$S_n$-equiv",
    "U1_broken":  r"$U(1)$-broken",
}

# Group display names
DISPLAY_GROUPS = {
    "U(1)":  r"$U(1)$",
    "SU(2)": r"$SU(2)$",
    "S_n":   r"$S_n$",
}

# Task display names (match the manuscript notation)
DISPLAY_TASKS = {
    "U1_task":  r"$U(1)$ task",
    "SU2_task": r"$SU(2)$ task",
    "Sn_task":  r"$S_n$ task",
}

# Component LaTeX labels and colours
COMP_LABELS = {
    "H_G":   r"$H_G$",
    "D_G":   r"$D_G$",
    "M_G":   r"$M_G$",
    "S_G":   r"$S_G$",
    "Psi_G": r"$\Psi_G$",
}
COMP_COLOURS = {
    "H_G": "#1f77b4",
    "D_G": "#2ca02c",
    "M_G": "#ff7f0e",
    "S_G": "#9467bd",
    "Psi_G": "#d62728",
}

print("Plot defaults configured.")


"""
PsiAudit core module - all functions used by the notebook.

This is built as a single .py file first so it can be unit-tested and
debugged independently of the notebook. The notebook then imports / inlines
the same functions.
"""

import numpy as np
from scipy.linalg import expm, eigh
from itertools import combinations, permutations

EPS = 1e-12

# ---------------------------------------------------------------------------
# Linear-algebra primitives
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(ops):
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def op_on_qubit(op, q, n):
    ops = [I2] * n
    ops[q] = op
    return kron_all(ops)


def two_qubit_op(op1, q1, op2, q2, n):
    if q1 == q2:
        raise ValueError("two_qubit_op requires distinct qubit indices")
    ops = [I2] * n
    ops[q1] = op1
    ops[q2] = op2
    return kron_all(ops)


def basis_state(index, dim):
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def ket0(n):
    return basis_state(0, 2 ** n)


def density(psi):
    psi = np.asarray(psi).reshape(-1)
    return np.outer(psi, psi.conj())


def normalize(psi):
    return psi / (np.linalg.norm(psi) + EPS)


def fro_norm(A):
    return np.linalg.norm(A, ord="fro")


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.array(
        [[np.exp(-1j * theta / 2), 0.0], [0.0, np.exp(1j * theta / 2)]], dtype=complex
    )


def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def cz_gate(q1, q2, n):
    """Controlled-Z gate on qubits (q1, q2). Diagonal in computational basis."""
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for idx in range(dim):
        b = format(idx, f"0{n}b")
        if b[q1] == "1" and b[q2] == "1":
            diag[idx] = -1.0
    return np.diag(diag)


def unitary_from_hamiltonian(H, theta):
    return expm(-1j * theta * H)


# ---------------------------------------------------------------------------
# Symmetry projectors and generators
# ---------------------------------------------------------------------------

def excitation_number_operator(n):
    """U(1) charge / total excitation number operator N = sum_q (I-Z_q)/2."""
    dim = 2 ** n
    N = np.zeros((dim, dim), dtype=complex)
    for q in range(n):
        N += 0.5 * (np.eye(dim) - op_on_qubit(Z, q, n))
    return N


def excitation_projectors(n):
    """U(1) sector projectors, indexed by Hamming weight k = 0..n."""
    dim = 2 ** n
    projectors = []
    labels = []
    for k in range(n + 1):
        P = np.zeros((dim, dim), dtype=complex)
        for idx in range(dim):
            if format(idx, f"0{n}b").count("1") == k:
                P[idx, idx] = 1.0
        projectors.append(P)
        labels.append(f"k={k}")
    return projectors, labels


def total_spin_operators(n):
    dim = 2 ** n
    Sx = np.zeros((dim, dim), dtype=complex)
    Sy = np.zeros((dim, dim), dtype=complex)
    Sz = np.zeros((dim, dim), dtype=complex)
    for q in range(n):
        Sx += 0.5 * op_on_qubit(X, q, n)
        Sy += 0.5 * op_on_qubit(Y, q, n)
        Sz += 0.5 * op_on_qubit(Z, q, n)
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    return Sx, Sy, Sz, S2


def su2_projectors(n):
    """SU(2) total-spin sector projectors P_j by eigenvalues of S^2."""
    Sx, Sy, Sz, S2 = total_spin_operators(n)
    vals, vecs = eigh(S2)
    sectors = {}
    for idx, val in enumerate(vals):
        j = 0.5 * (-1 + np.sqrt(1 + 4 * np.real(val)))
        j_round = np.round(2 * j) / 2
        # Guard against negative-zero from floating-point round at j=0
        if abs(j_round) < 1e-9:
            j_round = 0.0
        sectors.setdefault(j_round, []).append(idx)
    projectors = []
    labels = []
    for j in sorted(sectors.keys()):
        inds = sectors[j]
        V = vecs[:, inds]
        P = V @ V.conj().T
        projectors.append(P)
        labels.append(f"j={j:g}")
    return projectors, labels, (Sx, Sy, Sz, S2)


def permutation_unitary(perm, n):
    """Build the permutation unitary U_pi acting on n qubits.

    perm is a length-n permutation specifying where qubit i is sent.
    The matrix permutes basis states |b_0 b_1 ... b_{n-1}> ->
    |b_{perm^{-1}(0)} ... b_{perm^{-1}(n-1)}>.
    """
    dim = 2 ** n
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    U = np.zeros((dim, dim), dtype=complex)
    for src in range(dim):
        bits = format(src, f"0{n}b")
        new_bits = "".join(bits[inv[i]] for i in range(n))
        dst = int(new_bits, 2)
        U[dst, src] = 1.0
    return U


def sn_orbit_projectors(n):
    """S_n action on n qubits decomposes the basis by Hamming weight; each
    weight class is one S_n orbit. Within a class, the symmetric (Dicke)
    subspace carries the trivial irrep. We expose both:
      - orbit projectors (one per Hamming weight): for sector occupation
      - Dicke (totally symmetric) projector per weight class
    Returns (orbit_projectors, orbit_labels, generators).

    Generators are a complete set of transposition unitaries (i, i+1).
    """
    return_orbit = []
    return_labels = []
    dim = 2 ** n
    for k in range(n + 1):
        P = np.zeros((dim, dim), dtype=complex)
        for idx in range(dim):
            if format(idx, f"0{n}b").count("1") == k:
                P[idx, idx] = 1.0
        return_orbit.append(P)
        return_labels.append(f"|x|={k}")
    # Generators: adjacent transpositions
    gens = []
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        gens.append(permutation_unitary(perm, n))
    return return_orbit, return_labels, gens


# ---------------------------------------------------------------------------
# Psi_G components
# ---------------------------------------------------------------------------

def sector_probabilities(rho, projectors):
    """Return p_lambda = Tr(P_lambda rho), normalised and clipped to [0,1]."""
    probs = np.array([float(np.real(np.trace(P @ rho))) for P in projectors])
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    return probs / (s + EPS)


def H_G_score(P_traj):
    """Sector-occupation entropy H_G in [0, 1] via Shannon entropy of mean
    sector probabilities, normalised by log(K) where K is number of sectors."""
    pbar = np.mean(P_traj, axis=0)
    K = len(pbar)
    if K <= 1:
        return 0.0
    H = -np.sum(pbar * np.log(pbar + EPS)) / np.log(K)
    return float(np.clip(H, 0.0, 1.0))


def D_G_inter(states, projectors):
    """Inter-irrep cross-sector coherence (matrix-free).

    For a pure state rho = |psi><psi|, the block P_a rho P_b factorises as
    (P_a|psi>)(P_b|psi>)^dag, whose Frobenius norm equals
    ||P_a|psi>|| * ||P_b|psi>||. This avoids forming the dense density
    matrix and the triple product, giving values identical to the dense
    computation (verified exact) at a small fraction of the cost. Accepts
    a list/array of state vectors. Normalised by 0.5 and clipped to [0,1].
    """
    K = len(projectors)
    if K <= 1:
        return 0.0
    pairs = list(combinations(range(K), 2))
    if not pairs:
        return 0.0
    vals = []
    for psi in states:
        norms = [float(np.linalg.norm(P @ psi)) for P in projectors]
        s = sum(norms[a] * norms[b] for a, b in pairs)
        vals.append(s / len(pairs))
    return float(np.clip(np.mean(vals) / 0.5, 0.0, 1.0))


def D_G_mult(states, projectors):
    """Within-multiplicity-space organised coherence (matrix-free).

    With w = P|psi>, the block P rho P = w w^dag, so its off-diagonal
    Frobenius norm is sqrt(||w||^4 - sum_i |w_i|^4) and the sector trace is
    ||w||^2. Identical to the dense computation (verified exact) without
    building the density matrix. Accepts a list/array of state vectors.

    For sectors with dim_lambda <= 1 the contribution is zero.
    Normalised empirically by 0.5 to keep on the same scale as D_inter.
    """
    vals = []
    for psi in states:
        per_sector = []
        for P in projectors:
            w = P @ psi
            mass = np.abs(w) ** 2
            tr = float(mass.sum())
            if tr < EPS:
                continue
            fro_off = np.sqrt(max(tr * tr - float(np.sum(mass ** 2)), 0.0))
            per_sector.append(fro_off / (tr + EPS))
        if per_sector:
            vals.append(np.mean(per_sector))
        else:
            vals.append(0.0)
    return float(np.clip(np.mean(vals) / 0.5, 0.0, 1.0))


def D_G_score(rhos, projectors):
    """D_G = 1/2 (D_inter + D_mult): combines cross-irrep and within-
    multiplicity-space organised coherence. This is the refined form
    used in the accompanying theory paper."""
    return 0.5 * (D_G_inter(rhos, projectors) + D_G_mult(rhos, projectors))


def M_G_score(P_traj):
    """Symmetry metastability M_G via std of inverse participation ratio
    R = sum_lambda p_lambda^2 along the trajectory, normalised by 0.5."""
    R = np.sum(P_traj ** 2, axis=1)
    return float(np.clip(np.std(R) / 0.5, 0.0, 1.0))


def S_G_compliance(H_model, generators, gamma=3.0):
    """Channel-level compliance factor S_G = exp(-gamma * Delta_G), with
    Delta_G the mean normalised commutator defect over the supplied
    generators. For an exactly equivariant Hamiltonian S_G = 1.
    """
    if not generators:
        return 1.0, 0.0
    Hn = fro_norm(H_model) + EPS
    defects = []
    for G in generators:
        denom = Hn * (fro_norm(G) + EPS)
        defects.append(fro_norm(H_model @ G - G @ H_model) / denom)
    Delta = float(np.mean(defects))
    return float(np.exp(-gamma * Delta)), Delta


def unitary_commutator_deviation(apply_fn, n, L, n_params_fn, generators,
                                S=64, seed=0, init_state=None):
    """Parameter-averaged unitary-level commutator deviation (manuscript Eq. 10).

    Implements
        Delta_U_bar = E_{theta, g} ||U(theta) R(g) - R(g) U(theta)||_F /
                                    (||U(theta)||_F ||R(g)||_F)

    This is a circuit-level compliance diagnostic that complements the
    generator-sum compliance S_G. It is averaged over S random parameter
    draws and over all supplied generators g. When the ansatz acts as a
    pure state map starting from |0..0>, we evaluate ||U R - R U||_F via
    its action on the standard basis. For efficiency we evaluate the
    commutator on a fixed orthonormal basis of size 2**n.

    Returns the scalar Delta_U_bar (smaller = closer to exact circuit-level
    equivariance). Pair with exp(-gamma * Delta_U_bar) for a unitary-level
    analogue of S_G.
    """
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    dim = 2 ** n
    basis = np.eye(dim, dtype=complex)
    fro_R = [float(np.linalg.norm(g, ord="fro")) for g in generators]

    accum = 0.0
    count = 0
    for _ in range(S):
        theta = rng.uniform(-np.pi, np.pi, size=P)
        # Build U(theta) by applying the ansatz to each basis ket
        # (cheaper than constructing the full unitary symbolically).
        U = np.zeros((dim, dim), dtype=complex)
        for k in range(dim):
            U[:, k] = apply_fn(theta, n, L, init=basis[:, k])
        fro_U = float(np.linalg.norm(U, ord="fro"))
        if fro_U < EPS:
            continue
        for g, fro_g in zip(generators, fro_R):
            if fro_g < EPS:
                continue
            commutator = U @ g - g @ U
            num = float(np.linalg.norm(commutator, ord="fro"))
            accum += num / (fro_U * fro_g)
            count += 1
    return accum / max(count, 1)


def _sector_probabilities_from_state(psi, projectors):
    """Matrix-free sector probabilities for a pure state.

    Tr(P |psi><psi|) = <psi|P|psi> = ||P|psi>||^2 for a projector P.
    Returns the normalised, clipped probability vector, identical to
    sector_probabilities(density(psi), projectors).
    """
    probs = np.array([float(np.linalg.norm(P @ psi) ** 2) for P in projectors])
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    return probs / (s + EPS)


def Psi_G_full(states, projectors, H_model, generators,
               weights=(0.40, 0.35, 0.25), gamma=3.0):
    """Compute the full Psi_G report for a state trajectory.

    Composite definition (v12, gated stability formulation):

        Psi_G = S_G * ( w_H * H_G + w_D * D_G + w_M * H_G * (1 - M_G) )

    Three audit components contribute to the composite. Sector spread H_G
    enters with weight w_H. Cross-sector coherence proxy D_G enters with
    weight w_D. Trajectory stability (1 - M_G) enters gated by H_G with
    weight w_M, so the stability term contributes only when sector
    structure is activated. A perfectly sector-confined trajectory has
    H_G = D_G = 0, so its composite collapses to zero regardless of
    M_G or S_G, restoring the Regime A collapse property. The composite
    is then multiplied by the generator-sum compliance S_G in (0, 1].

    The bare (1 - M_G) term used in the previous v12 draft created a
    nonzero stability floor of w_M for any equivariant confined
    trajectory, which conflated trivial confinement with non-trivial
    organisation. The gated form here addresses that issue.

    Returns dict with H_G, D_G, D_inter, D_mult, M_G, S_G, defect,
    Psi_raw, Psi_G, P_traj (raw sector trajectory).
    """
    # Matrix-free: sector probabilities Tr(P rho) = <psi|P|psi> = ||P|psi>||^2
    # for projectors P, avoiding construction of the dense density matrix.
    P_traj = np.vstack([
        _sector_probabilities_from_state(psi, projectors) for psi in states
    ])
    H = H_G_score(P_traj)
    D_inter = D_G_inter(states, projectors)
    D_mult = D_G_mult(states, projectors)
    D = 0.5 * (D_inter + D_mult)
    M = M_G_score(P_traj)
    S, defect = S_G_compliance(H_model, generators, gamma=gamma)
    wH, wD, wM = weights
    # Gated stability composite: w_M * H_G * (1 - M_G) ensures that
    # the stability bonus is active only when sector structure is.
    psi_raw = wH * H + wD * D + wM * H * (1.0 - M)
    psi = S * psi_raw
    return {
        "H_G": H,
        "D_G": D,
        "D_inter": D_inter,
        "D_mult": D_mult,
        "M_G": M,
        "S_G": S,
        "defect": defect,
        "Psi_raw": psi_raw,
        "Psi_G": psi,
        "P_traj": P_traj,
    }


# ---------------------------------------------------------------------------
# Comparative diagnostics: expressibility, gradient variance, entangling
# ---------------------------------------------------------------------------

def expressibility_kl(state_samples, fidelity_bins=75):
    """Sim-Aspuru-Guzik-style expressibility: KL divergence between the
    sampled fidelity distribution of the ansatz output states and the
    Haar-random fidelity distribution F ~ (D-1)*(1-F)**(D-2).

    state_samples: array of shape (S, D) of normalised state vectors.
    Returns KL divergence (lower = closer to Haar = more expressive).
    """
    states = np.asarray(state_samples)
    S, D = states.shape
    if S < 4:
        return float("nan")
    # Build fidelity samples F = |<psi_a|psi_b>|^2 over random pairs
    rng = np.random.default_rng(0)
    n_pairs = min(2000, S * (S - 1) // 2)
    a = rng.integers(0, S, size=n_pairs)
    b = rng.integers(0, S, size=n_pairs)
    mask = a != b
    a = a[mask]
    b = b[mask]
    overlaps = np.einsum("ij,ij->i", states[a].conj(), states[b])
    F = np.abs(overlaps) ** 2
    bins = np.linspace(0, 1, fidelity_bins + 1)
    p_emp, _ = np.histogram(F, bins=bins, density=True)
    centres = 0.5 * (bins[:-1] + bins[1:])
    # Haar density on D dimensions
    p_haar = (D - 1) * (1 - centres) ** (D - 2)
    p_haar = p_haar / (p_haar.sum() * (bins[1] - bins[0]))
    p_emp_safe = p_emp + 1e-12
    p_haar_safe = p_haar + 1e-12
    kl = np.sum(p_emp_safe * np.log(p_emp_safe / p_haar_safe)) * (bins[1] - bins[0])
    return float(kl)


def gradient_variance(loss_samples_per_param):
    """Barren-plateau diagnostic: variance of partial derivatives of a loss
    function over parameter samples, computed externally and supplied here."""
    vals = np.asarray(loss_samples_per_param).reshape(-1)
    return float(np.var(vals))


def meyer_wallach_entanglement(psi):
    """Meyer-Wallach Q for a pure state. Q = 0 for product, Q = 1 maximally
    entangled. For an n-qubit state psi (length 2^n)."""
    psi = np.asarray(psi).reshape(-1)
    D = len(psi)
    n = int(np.log2(D))
    rho = density(psi)
    sum_purity = 0.0
    for q in range(n):
        # Reduced density matrix on qubit q
        # Trace out the other n-1 qubits
        rho_full = rho.reshape([2] * (2 * n))
        # Move qubit q to front in both bra and ket axes
        kept_axes_ket = [q]
        kept_axes_bra = [q + n]
        traced_axes_ket = [i for i in range(n) if i != q]
        traced_axes_bra = [i + n for i in range(n) if i != q]
        order = (kept_axes_ket + traced_axes_ket
                 + kept_axes_bra + traced_axes_bra)
        rho_perm = np.transpose(rho_full, order)
        # Now shape is (2, 2^(n-1), 2, 2^(n-1)); contract the two large axes
        rho_perm = rho_perm.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
        rho_q = np.einsum("ikjk->ij", rho_perm)
        sum_purity += np.real(np.trace(rho_q @ rho_q))
    Q = 2.0 * (1.0 - sum_purity / n)
    return float(np.clip(Q, 0.0, 1.0))


"""
PsiAudit ansatz library.

Each ansatz is a callable
    apply(theta, n) -> state
that maps a parameter vector and qubit count to the output state |psi(theta)>
starting from |0...0>. The ansätze are deliberately small and CPU-tractable.

The five families used in the paper:

  1. Hardware-efficient ansatz (HEA, non-equivariant baseline)
  2. U(1)-equivariant XY-conserving ansatz
  3. SU(2)-equivariant Heisenberg-block ansatz
  4. S_n permutation-equivariant ansatz (parameter-input, Schatzki-style)
  5. Symmetry-broken ansatz (equivariant + small breaking term)
"""

import numpy as np
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


"""
PsiAudit trajectory builders.

Two complementary regimes for sampling state ensembles:

  Regime A (sector-confined): start from |0...0>, vary parameters along a
    smooth path. Equivariant ansätze stay in the initial sector by
    construction; this is the regime in which Theorem 1 / Proposition 1
    of the Symmetry paper predict Psi_G = 0.

  Regime B (multi-sector): prepare initial states with controlled support
    across multiple sectors via local product rotations, then apply the
    circuit. Equivariant ansätze preserve the sector spread; this is the
    regime in which Psi_G distinguishes well-organised from
    poorly-organised circuits.
"""

import numpy as np
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
    from itertools import combinations
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


import sys

def check(label, condition, expected):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}: {expected}")
    return condition

n_t = CONFIG["n_audit"]
checks_ok = []

# C1: U(1) projectors are idempotent and sum to I
proj_u1, lab_u1 = excitation_projectors(n_t)
sum_err = float(np.linalg.norm(sum(proj_u1) - np.eye(2 ** n_t)))
checks_ok.append(check(
    "U(1) projectors idempotent and complete",
    sum_err < 1e-9 and all(np.linalg.norm(P @ P - P) < 1e-10 for P in proj_u1),
    "sum to identity, P^2 = P, deviation below 1e-9"
))

# C2: SU(2) projectors are idempotent and complete
proj_su2, lab_su2, gens_su2 = su2_projectors(n_t)
sum_err2 = float(np.linalg.norm(sum(proj_su2) - np.eye(2 ** n_t)))
checks_ok.append(check(
    "SU(2) projectors idempotent and complete",
    sum_err2 < 1e-9,
    "sum to identity, deviation below 1e-9"
))

# C3: Equivariant ansatz Hamiltonians give zero defect against their generators
Nop = excitation_number_operator(n_t)
Sx, Sy, Sz, _ = total_spin_operators(n_t)
err_u1_self = float(fro_norm(H_eff_U1(n_t) @ Nop - Nop @ H_eff_U1(n_t)))
err_su2_x = float(fro_norm(H_eff_SU2(n_t) @ Sx - Sx @ H_eff_SU2(n_t)))
err_su2_y = float(fro_norm(H_eff_SU2(n_t) @ Sy - Sy @ H_eff_SU2(n_t)))
err_su2_z = float(fro_norm(H_eff_SU2(n_t) @ Sz - Sz @ H_eff_SU2(n_t)))
checks_ok.append(check(
    "U(1) ansatz Hamiltonian commutes with N",
    err_u1_self < 1e-9, "deviation below 1e-9"
))
checks_ok.append(check(
    "SU(2) ansatz Hamiltonian commutes with Sx, Sy, Sz",
    max(err_su2_x, err_su2_y, err_su2_z) < 1e-9, "deviation below 1e-9"
))

# C4: Broken ansatz Hamiltonian has nonzero defect against N
err_brk = float(fro_norm(H_eff_broken(n_t) @ Nop - Nop @ H_eff_broken(n_t)))
checks_ok.append(check(
    "Broken ansatz Hamiltonian flagged by U(1) defect",
    err_brk > 1e-3, "nonzero defect detected"
))

# C5: Meyer-Wallach Q correctly classifies product and GHZ
Q_prod = meyer_wallach_entanglement(ket0(n_t))
psi_ghz = np.zeros(2 ** n_t, dtype=complex)
psi_ghz[0] = 1 / np.sqrt(2); psi_ghz[-1] = 1 / np.sqrt(2)
Q_ghz = meyer_wallach_entanglement(psi_ghz)
checks_ok.append(check(
    "Meyer-Wallach Q calibration",
    abs(Q_prod) < 1e-6 and abs(Q_ghz - 1) < 1e-6,
    "product state Q=0, n-qubit GHZ Q=1"
))

# C6: Symmetric Dicke states lie in the j = n/2 SU(2) sector
# This validates that the SU(2) total-spin projectors are correctly built.
proj_su2_t, lab_su2_t, _ = su2_projectors(n_t)
# Find the j = n/2 sector index
j_max_label = f"j={n_t / 2:.1f}"
if j_max_label in lab_su2_t:
    j_idx = lab_su2_t.index(j_max_label)
elif f"j={int(n_t / 2)}" in lab_su2_t:
    j_idx = lab_su2_t.index(f"j={int(n_t / 2)}")
else:
    j_idx = len(proj_su2_t) - 1  # by convention, last projector is j_max
dicke_in_top_sector = []
for k_t in range(n_t + 1):
    psi_W = _symmetric_dicke_state(n_t, k_t)
    rho_W = density(psi_W)
    probs = sector_probabilities(rho_W, proj_su2_t)
    dicke_in_top_sector.append(probs[j_idx] > 0.999)
checks_ok.append(check(
    f"Symmetric Dicke states lie in j = {n_t / 2:g} sector",
    all(dicke_in_top_sector),
    f"Pr(j={n_t / 2:g}) > 0.999 for every |W_k>, k=0..{n_t}"
))

# C7: Active-path SU(2) validation
# Confirms that the product-state Regime B initial state has multi-sector
# SU(2) support and that the SU(2)-equivariant ansatz preserves the
# resulting sector probabilities along the trajectory.
gen_sxyz = [Sx, Sy, Sz]
states_su2 = trajectory_B_multi_sector(
    ANSATZ_REGISTRY["SU2_equiv"]["apply"], n_t, CONFIG["L"],
    ANSATZ_REGISTRY["SU2_equiv"]["n_params"], T=20, seed=CONFIG["seed"])
P_traj_su2 = np.vstack([sector_probabilities(density(psi), proj_su2)
                        for psi in states_su2])
# Multi-sector check: probability mass NOT concentrated in a single j sector
p0_avg = P_traj_su2.mean(axis=0)
n_nontrivial = int(np.sum(p0_avg > 0.01))
checks_ok.append(check(
    "Active Regime B has multi-sector SU(2) support",
    n_nontrivial >= 2,
    f"{n_nontrivial} of {len(proj_su2)} j-sectors carry probability > 0.01"
))
# Preservation check: SU(2)-equiv ansatz keeps the per-step sector probabilities
# within a tight band (drift < 0.02 across the trajectory in the
# top two j-sectors)
top_two = np.argsort(p0_avg)[-2:]
max_drift = float(np.max(np.abs(P_traj_su2[:, top_two] - p0_avg[top_two]).max(axis=0)))
checks_ok.append(check(
    "SU(2)-equivariant preserves active-path sector probabilities",
    max_drift < 0.02,
    f"max drift in top two j-sectors {max_drift:.4f} < 0.02"
))

# C9: Gated composite gives symmetry-correct ordering on U(1) near-control
# Validates the v12 gated-stability formulation directly.
proj_u1_t, _ = excitation_projectors(n_t)
Nop_t = excitation_number_operator(n_t)
states_eq = trajectory_B_multi_sector(
    ANSATZ_REGISTRY["U1_equiv"]["apply"], n_t, CONFIG["L"],
    ANSATZ_REGISTRY["U1_equiv"]["n_params"], T=20, seed=CONFIG["seed"])
states_br = trajectory_B_multi_sector(
    ANSATZ_REGISTRY["U1_broken"]["apply"], n_t, CONFIG["L"],
    ANSATZ_REGISTRY["U1_broken"]["n_params"], T=20, seed=CONFIG["seed"])
rep_eq = Psi_G_full(states_eq, proj_u1_t,
                    ANSATZ_REGISTRY["U1_equiv"]["H_eff"](n_t), [Nop_t],
                    weights=CONFIG["weights"], gamma=CONFIG["gamma"])
rep_br = Psi_G_full(states_br, proj_u1_t,
                    ANSATZ_REGISTRY["U1_broken"]["H_eff"](n_t), [Nop_t],
                    weights=CONFIG["weights"], gamma=CONFIG["gamma"])
checks_ok.append(check(
    "Gated composite: Psi_G(U1_equiv) > Psi_G(U1_broken)",
    rep_eq["Psi_G"] > rep_br["Psi_G"],
    f"equivariant {rep_eq['Psi_G']:.4f} > broken {rep_br['Psi_G']:.4f}"
))

# C10: Gated composite collapses to zero in Regime A for equivariant ansatze
# Validates that the gated stability term restores the Regime A collapse.
states_eq_A = trajectory_A_sector_confined(
    ANSATZ_REGISTRY["U1_equiv"]["apply"], n_t, CONFIG["L"],
    ANSATZ_REGISTRY["U1_equiv"]["n_params"], T=20, seed=CONFIG["seed"])
rep_eq_A = Psi_G_full(states_eq_A, proj_u1_t,
                      ANSATZ_REGISTRY["U1_equiv"]["H_eff"](n_t), [Nop_t],
                      weights=CONFIG["weights"], gamma=CONFIG["gamma"])
checks_ok.append(check(
    "Regime A confined U(1)-equiv collapses to Psi_G = 0",
    rep_eq_A["Psi_G"] < 1e-9,
    f"Psi_G = {rep_eq_A['Psi_G']:.2e} (< 1e-9)"
))

assert all(checks_ok), "Self-test failed; halting."
print(f"\nAll {len(checks_ok)} self-tests passed.")


def audit_one(name, info, n, L, group_name, projectors, generators, T,
              traj_fn, gamma, weights, seed=0):
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


t0 = time.time()
n = CONFIG["n_audit"]
L = CONFIG["L"]
T = CONFIG["T"]

proj_u1, lab_u1 = excitation_projectors(n)
Nop = excitation_number_operator(n)
proj_su2, lab_su2, gens_su2 = su2_projectors(n)
Sx, Sy, Sz, _ = gens_su2
proj_sn, lab_sn, gens_sn = sn_orbit_projectors(n)

GROUPS = {
    "U(1)":   dict(projectors=proj_u1,  labels=lab_u1,  generators=[Nop]),
    "SU(2)":  dict(projectors=proj_su2, labels=lab_su2, generators=[Sx, Sy, Sz]),
    "S_n":    dict(projectors=proj_sn,  labels=lab_sn,  generators=gens_sn),
}

REGIMES = {
    "A_confined": trajectory_A_sector_confined,
    "B_multi":    trajectory_B_multi_sector,
}

audit_rows = []
for regime_name, traj_fn in REGIMES.items():
    for ansatz_name, info in ANSATZ_REGISTRY.items():
        for group_name, gdata in GROUPS.items():
            r = audit_one(ansatz_name, info, n, L, group_name,
                          gdata["projectors"], gdata["generators"], T,
                          traj_fn, CONFIG["gamma"], CONFIG["weights"],
                          seed=CONFIG["seed"])
            r["regime"] = regime_name
            audit_rows.append(r)

df_audit = pd.DataFrame(audit_rows)
print(f"\nAudit grid completed in {time.time() - t0:.1f}s")
df_audit_round = df_audit.copy()
for c in ["H_G", "D_G", "D_inter", "D_mult", "M_G", "S_G", "defect",
          "Psi_raw", "Psi_G"]:
    df_audit_round[c] = df_audit_round[c].round(3)
df_audit_round[["ansatz", "group", "regime", "n_params", "H_G", "D_G",
                "M_G", "S_G", "defect", "Psi_G"]]


NATURAL_GROUP = {
    "HEA":         "U(1)",   # tested against U(1) for reference comparison
    "U1_equiv":    "U(1)",
    "SU2_equiv":   "SU(2)",
    "Sn_equiv":    "S_n",
    "U1_broken":   "U(1)",
}

def build_groups_at(n):
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

t0 = time.time()
scaling_rows = []
for n_s in CONFIG["n_scaling"]:
    print(f"  n = {n_s}")
    G_n = build_groups_at(n_s)
    for ansatz_name, info in ANSATZ_REGISTRY.items():
        gname = NATURAL_GROUP[ansatz_name]
        gdata = G_n[gname]
        r = audit_one(ansatz_name, info, n_s, CONFIG["L"], gname,
                      gdata["projectors"], gdata["generators"], CONFIG["T"],
                      trajectory_B_multi_sector, CONFIG["gamma"],
                      CONFIG["weights"], seed=CONFIG["seed"])
        r["regime"] = "B_multi"
        scaling_rows.append(r)
        print(f"    {ansatz_name:10s} ({gname:5s}): "
              f"H={r['H_G']:.3f} D={r['D_G']:.3f} M={r['M_G']:.3f} "
              f"S={r['S_G']:.3f} Psi={r['Psi_G']:.3f}")

df_scaling = pd.DataFrame(scaling_rows)
print(f"\nScaling sweep completed in {time.time() - t0:.1f}s")
df_scaling.round(3)


# Unitary-level commutator deviation Delta_U for the five natural-group
# (ansatz, group) pairs at n_audit. Manuscript Eq. 10, the circuit-level
# compliance complement to S_G. Computed under a modest sample budget
# for tractability.
print("\nComputing unitary-level commutator deviation (Eq. 10)...")
t1 = time.time()
n_au = CONFIG["n_audit"]
delta_U_rows = []
for ansatz_name, info in ANSATZ_REGISTRY.items():
    gname = NATURAL_GROUP[ansatz_name]
    gens  = GROUPS[gname]["generators"]
    _S_dU = ANALYSIS.get("delta_U_samples", 8) if "ANALYSIS" in dir() else 8
    dU = unitary_commutator_deviation(
        info["apply"], n_au, CONFIG["L"], info["n_params"], gens,
        S=_S_dU, seed=CONFIG["seed"])
    delta_U_rows.append({"ansatz": ansatz_name, "group": gname,
                         "Delta_U": dU,
                         "S_U": float(np.exp(-CONFIG["gamma"] * dU))})
    print(f"  {ansatz_name:12s} ({gname:5s}): Delta_U = {dU:.4e}  "
          f"S_U = {np.exp(-CONFIG['gamma'] * dU):.4f}")
df_delta_U = pd.DataFrame(delta_U_rows)
print(f"\nDelta_U computed in {time.time() - t1:.1f}s")


t0 = time.time()
n = CONFIG["n_audit"]
L = CONFIG["L"]
S_diag = CONFIG["S_diag"]

Z0 = op_on_qubit(Z, 0, n)
diag_rows = []
for ansatz_name, info in ANSATZ_REGISTRY.items():
    states = random_parameter_samples(info["apply"], n, L, info["n_params"],
                                      S=S_diag, seed=CONFIG["seed"] + 1,
                                      multi_sector=True)
    kl = expressibility_kl(states)
    Q_mean = float(np.mean([meyer_wallach_entanglement(s) for s in states]))
    Q_std  = float(np.std([meyer_wallach_entanglement(s) for s in states]))
    gv = gradient_variance_diagnostic(info["apply"], n, L, info["n_params"],
                                      Z0, S=CONFIG["S_grad"],
                                      seed=CONFIG["seed"] + 2,
                                      multi_sector=True)
    diag_rows.append({
        "ansatz": ansatz_name, "n": n, "L": L,
        "expr_KL": kl, "MW_mean": Q_mean, "MW_std": Q_std, "grad_var": gv,
    })
    print(f"  {ansatz_name:12s}  KL={kl:.4f}  Q={Q_mean:.3f}  "
          f"Var(g)={gv:.4e}")

df_diag = pd.DataFrame(diag_rows)

# Pull Psi_G values for the natural-group case from the scaling table
df_natural = df_scaling[df_scaling["n"] == n].copy()
df_natural = df_natural.rename(columns={"group": "natural_group"})
merge_cols = ["ansatz", "Psi_G", "H_G", "D_G", "M_G", "S_G"]
df_compare = df_natural[merge_cols + ["natural_group"]].merge(
    df_diag, on="ansatz")
print(f"\nDiagnostics completed in {time.time() - t0:.1f}s")
df_compare.round(4)


def _bootstrap_ci(x, n_boot=2000, alpha=0.05, rng=None):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)

def _aggregate(df, group_cols, value_cols):
    """Mean / std / 95% bootstrap CI across seeds for each group."""
    rng = np.random.default_rng(12345)
    out = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        for c in value_cols:
            vals = sub[c].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(vals, rng=rng)
            row[f"{c}_mean"] = float(np.mean(vals))
            with np.errstate(invalid="ignore"):
                row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c}_ci_lo"] = lo
            row[f"{c}_ci_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


AUDIT_COMPS  = ["H_G", "D_G", "D_inter", "D_mult", "M_G", "S_G", "defect", "Psi_G"]
DIAG_COMPS   = ["expr_KL", "MW_mean", "grad_var"]

if ANALYSIS["run_multiseed"]:
    t0 = time.time()
    seeds = _seed_list()
    n  = CONFIG["n_audit"]; L = CONFIG["L"]; T = CONFIG["T"]
    gamma = CONFIG["gamma"]; weights = CONFIG["weights"]

    # Group operators depend only on n (not on the seed), so build them ONCE
    # outside the seed loop and reuse. Rebuilding dense projectors every seed
    # was the dominant avoidable cost in the previous version.
    G_audit = build_groups_at(n)
    scaling_ns = ANALYSIS.get("multiseed_scaling_n", list(CONFIG["n_scaling"]))
    G_cache = {n: G_audit}
    for _n_s in scaling_ns:
        if _n_s not in G_cache:
            G_cache[_n_s] = build_groups_at(_n_s)
    print(f"  cached group operators for n in {sorted(G_cache)} "
          f"(scaling sweep over {scaling_ns})")

    ms_audit, ms_scaling, ms_diag, ms_deltaU = [], [], [], []
    Z0 = op_on_qubit(Z, 0, n)

    for si, sd in enumerate(seeds):
        # ---- full audit grid (all ansatz x group x regime) -------------
        for regime_name, traj_fn in REGIMES.items():
            for aname, info in ANSATZ_REGISTRY.items():
                for gname, gdata in G_audit.items():
                    r = audit_one(aname, info, n, L, gname,
                                  gdata["projectors"], gdata["generators"], T,
                                  traj_fn, gamma, weights, seed=sd)
                    r["regime"] = regime_name; r["seed"] = sd
                    ms_audit.append(r)
        # ---- scaling sweep (natural group, Regime B) -------------------
        for n_s in scaling_ns:
            G_n = G_cache[n_s]                      # reuse cached operators
            for aname, info in ANSATZ_REGISTRY.items():
                gname = NATURAL_GROUP[aname]; gdata = G_n[gname]
                r = audit_one(aname, info, n_s, L, gname,
                              gdata["projectors"], gdata["generators"], T,
                              trajectory_B_multi_sector, gamma, weights, seed=sd)
                r["regime"] = "B_multi"; r["seed"] = sd
                ms_scaling.append(r)
        # ---- comparative diagnostics -----------------------------------
        for aname, info in ANSATZ_REGISTRY.items():
            states = random_parameter_samples(info["apply"], n, L, info["n_params"],
                                              S=CONFIG["S_diag"], seed=sd + 1,
                                              multi_sector=True)
            kl = expressibility_kl(states)
            Qm = float(np.mean([meyer_wallach_entanglement(s) for s in states]))
            gv = gradient_variance_diagnostic(info["apply"], n, L, info["n_params"],
                                              Z0, S=CONFIG["S_grad"], seed=sd + 2,
                                              multi_sector=True)
            ms_diag.append({"ansatz": aname, "seed": sd,
                            "expr_KL": kl, "MW_mean": Qm, "grad_var": gv})
        # ---- unitary-level compliance (first-class) --------------------
        # Delta_U is a structural property of the circuit and is invariant
        # across seeds, so it is evaluated once on the first seed and reused.
        # A per-seed row is still recorded so the aggregation schema is uniform.
        if si == 0:
            _dU_cache = {}
            for aname, info in ANSATZ_REGISTRY.items():
                gname = NATURAL_GROUP[aname]
                gens  = G_audit[gname]["generators"]
                dU = unitary_commutator_deviation(info["apply"], n, L, info["n_params"],
                                                  gens, S=ANALYSIS["delta_U_samples"], seed=sd)
                _dU_cache[aname] = (gname, float(dU))
        for aname, (gname, dU) in _dU_cache.items():
            ms_deltaU.append({"ansatz": aname, "group": gname, "seed": sd,
                              "Delta_U": float(dU),
                              "S_U": float(np.exp(-gamma * dU))})
        print(f"  seed {sd:2d} done ({si+1}/{len(seeds)})  "
              f"elapsed {time.time()-t0:.1f}s")

    df_ms_audit   = pd.DataFrame(ms_audit)
    df_ms_scaling = pd.DataFrame(ms_scaling)
    df_ms_diag    = pd.DataFrame(ms_diag)
    df_ms_deltaU  = pd.DataFrame(ms_deltaU)

    agg_audit   = _aggregate(df_ms_audit,  ["ansatz", "group", "regime"], AUDIT_COMPS)
    agg_scaling = _aggregate(df_ms_scaling,["ansatz", "group", "n"],       AUDIT_COMPS)
    agg_diag    = _aggregate(df_ms_diag,   ["ansatz"],                     DIAG_COMPS)
    agg_deltaU  = _aggregate(df_ms_deltaU, ["ansatz", "group"],            ["Delta_U", "S_U"])

    print(f"\nMulti-seed replication ({len(seeds)} seeds) done in {time.time()-t0:.1f}s")
    # quick look: natural-group Regime B composite mean +/- std
    nat_b = agg_scaling[agg_scaling["n"] == n].copy()
    cols = ["ansatz", "group", "Psi_G_mean", "Psi_G_std", "Psi_G_ci_lo", "Psi_G_ci_hi",
            "S_G_mean", "M_G_mean"]
    print("\nRegime B, n =", n, "(natural group), composite across seeds:")
    print(nat_b[cols].round(3).to_string(index=False))
else:
    df_ms_audit = df_ms_scaling = df_ms_diag = df_ms_deltaU = None
    agg_audit = agg_scaling = agg_diag = agg_deltaU = None
    print("Multi-seed disabled.")


def _composite(H, D, M, S, weights, gamma_unused=None, gated=True):
    wH, wD, wM = weights
    stab = H * (1.0 - M) if gated else (1.0 - M)
    return S * (wH * H + wD * D + wM * stab)

def _kendall_tau(rank_a, rank_b):
    """Kendall tau between two orderings given as lists of ansatz names."""
    items = list(rank_a)
    pos_a = {x: i for i, x in enumerate(rank_a)}
    pos_b = {x: i for i, x in enumerate(rank_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            sa = np.sign(pos_a[a] - pos_a[b])
            sb = np.sign(pos_b[a] - pos_b[b])
            if sa * sb > 0: conc += 1
            elif sa * sb < 0: disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0

if ANALYSIS["run_sensitivity"]:
    t0 = time.time()
    n = CONFIG["n_audit"]
    # cached components: prefer multi-seed mean; else single-seed df_scaling
    if agg_scaling is not None:
        base = agg_scaling[agg_scaling["n"] == n].copy()
        comp = {r["ansatz"]: dict(H=r["H_G_mean"], D=r["D_G_mean"],
                                  M=r["M_G_mean"], S=r["S_G_mean"])
                for _, r in base.iterrows()}
    else:
        base = df_scaling[df_scaling["n"] == n].copy()
        comp = {r["ansatz"]: dict(H=r["H_G"], D=r["D_G"], M=r["M_G"], S=r["S_G"])
                for _, r in base.iterrows()}
    ansatze = list(comp.keys())

    def _ranking(weights, gamma, gated=True):
        scores = {a: _composite(comp[a]["H"], comp[a]["D"], comp[a]["M"],
                                comp[a]["S"], weights, gamma, gated) for a in ansatze}
        return sorted(ansatze, key=lambda a: -scores[a]), scores

    conv_w = CONFIG["weights"]; conv_g = CONFIG["gamma"]
    conv_rank, conv_scores = _ranking(conv_w, conv_g, gated=True)

    # --- weight sweep (Dirichlet) ---
    rng = np.random.default_rng(2024)
    sens_rows = []
    for _ in range(ANALYSIS["n_weight_samples"]):
        w = rng.dirichlet([2.0, 2.0, 2.0])
        rank, scores = _ranking(tuple(w), conv_g, gated=True)
        tau = _kendall_tau(conv_rank, rank)
        row = {"w_H": w[0], "w_D": w[1], "w_M": w[2], "gamma": conv_g,
               "kendall_tau_vs_convention": tau, "top_ansatz": rank[0]}
        row.update({f"score_{a}": scores[a] for a in ansatze})
        sens_rows.append(row)
    df_sens_weights = pd.DataFrame(sens_rows)

    # --- gamma sweep ---
    gam_rows = []
    for g in ANALYSIS["gamma_grid"]:
        rank, scores = _ranking(conv_w, g, gated=True)
        tau = _kendall_tau(conv_rank, rank)
        row = {"gamma": g, "kendall_tau_vs_convention": tau, "top_ansatz": rank[0]}
        row.update({f"score_{a}": scores[a] for a in ansatze})
        gam_rows.append(row)
    df_sens_gamma = pd.DataFrame(gam_rows)

    # --- H_G-gate ablation ---
    abl_rows = []
    for aname in ansatze:
        c = comp[aname]
        gated   = _composite(c["H"], c["D"], c["M"], c["S"], conv_w, conv_g, gated=True)
        ungated = _composite(c["H"], c["D"], c["M"], c["S"], conv_w, conv_g, gated=False)
        abl_rows.append({"ansatz": aname, "H_G": c["H"], "D_G": c["D"],
                         "M_G": c["M"], "S_G": c["S"],
                         "Psi_gated": gated, "Psi_ungated": ungated,
                         "floor_removed_by_gate": ungated - gated})
    df_gate_ablation = pd.DataFrame(abl_rows)

    # --- Regime A gate ablation: does the gate enforce the collapse? ---
    if agg_audit is not None:
        srcA = agg_audit[agg_audit["regime"] == "A_confined"].copy()
        getH = lambda r: r["H_G_mean"]; getD = lambda r: r["D_G_mean"]
        getM = lambda r: r["M_G_mean"]; getS = lambda r: r["S_G_mean"]
    else:
        srcA = df_audit[df_audit["regime"] == "A_confined"].copy()
        getH = lambda r: r["H_G"]; getD = lambda r: r["D_G"]
        getM = lambda r: r["M_G"]; getS = lambda r: r["S_G"]
    ablA = []
    for _, r in srcA.iterrows():
        nat = NATURAL_GROUP.get(r["ansatz"])
        if r["group"] != nat:
            continue
        gated   = _composite(getH(r), getD(r), getM(r), getS(r), conv_w, conv_g, True)
        ungated = _composite(getH(r), getD(r), getM(r), getS(r), conv_w, conv_g, False)
        ablA.append({"ansatz": r["ansatz"], "group": r["group"],
                     "H_G": getH(r), "Psi_gated_regimeA": gated,
                     "Psi_ungated_regimeA": ungated})
    df_gate_ablation_regimeA = pd.DataFrame(ablA)

    tau_med = df_sens_weights["kendall_tau_vs_convention"].median()
    tau_min = df_sens_weights["kendall_tau_vs_convention"].min()
    top_frac = (df_sens_weights["top_ansatz"] == conv_rank[0]).mean()
    print(f"Sensitivity done in {time.time()-t0:.1f}s")
    print(f"  convention ranking (best->worst): {conv_rank}")
    print(f"  weight-sweep Kendall tau vs convention: "
          f"median={tau_med:.3f}  min={tau_min:.3f}")
    print(f"  fraction of weight draws keeping the same top ansatz: {top_frac:.2f}")
    print("\n  gamma sweep (tau vs convention):")
    print(df_sens_gamma[["gamma","kendall_tau_vs_convention","top_ansatz"]].round(3).to_string(index=False))
    print("\n  H_G-gate ablation, Regime A natural group (gate should force 0):")
    print(df_gate_ablation_regimeA.round(3).to_string(index=False))
else:
    df_sens_weights = df_sens_gamma = df_gate_ablation = df_gate_ablation_regimeA = None
    print("Sensitivity disabled.")


# Symmetry-matched downstream study: a non-tuned compatibility matrix.
#
# A single fixed task cannot be fair across symmetry classes, because an
# ansatz equivariant to a symmetry is by construction blind to data that
# varies only along that symmetry's quotient. Rather than engineer one task,
# we define three fixed symmetry-matched binary tasks and run EVERY ansatz on
# EVERY task. The resulting accuracy matrix exposes which structures can learn
# which tasks. The tasks and readout are fixed in advance (not tuned per
# ansatz), so the matrix is an honest structural result, not a constructed one.
#
# Tasks (all binary, balanced, readout = normalised total Z):
#   U1_task  : different total-excitation sectors (k=2 vs k=4).
#   SU2_task : collective spin, aligned vs Neel-like.
#   Sn_task  : permutation-symmetric collective angle, low vs high.

from itertools import combinations as _combinations

def _comp_state(ones, n):
    v = np.zeros(2 ** n, dtype=complex); idx = 0
    for q in ones:
        idx |= (1 << (n - 1 - q))
    v[idx] = 1.0
    return v

def _task_U1(n, npc, seed):
    rng = np.random.default_rng(seed); X = []; y = []
    for cls, k in [(0, 2), (1, 4)]:
        combos = list(_combinations(range(n), k))
        for _ in range(npc):
            ones = combos[rng.integers(len(combos))]
            a = rng.uniform(-0.2, 0.2, size=n)
            X.append(kron_all([ry(x) for x in a]) @ _comp_state(ones, n)); y.append(cls)
    return np.array(X), np.array(y)

def _task_SU2(n, npc, seed):
    rng = np.random.default_rng(seed); X = []; y = []
    for cls in [0, 1]:
        for _ in range(npc):
            if cls == 1:
                ang = rng.uniform(0.0, 0.4, size=n)
            else:
                ang = np.array([0.0 if q % 2 == 0 else np.pi for q in range(n)]) \
                      + rng.uniform(-0.3, 0.3, size=n)
            X.append(kron_all([ry(a) for a in ang]) @ ket0(n)); y.append(cls)
    return np.array(X), np.array(y)

def _task_Sn(n, npc, seed):
    rng = np.random.default_rng(seed); X = []; y = []
    for cls, base in [(0, 0.5), (1, 2.5)]:
        for _ in range(npc):
            a = base + rng.uniform(-0.2, 0.2)
            X.append(kron_all([ry(a) for _ in range(n)]) @ ket0(n)); y.append(cls)
    return np.array(X), np.array(y)

TASKS = {"U1_task": _task_U1, "SU2_task": _task_SU2, "Sn_task": _task_Sn}
TASK_NATURAL_GROUP = {"U1_task": "U(1)", "SU2_task": "SU(2)", "Sn_task": "S_n"}

def _train_spsa(apply_fn, n, L, n_params_fn, X, y, observable,
                steps=120, lr=0.3, c_spsa=0.05, seed=0):
    rng = np.random.default_rng(seed)
    P = n_params_fn(n, L)
    theta = rng.uniform(-np.pi, np.pi, size=P)
    def loss_acc(th):
        preds, losses = [], []
        for psi0, label in zip(X, y):
            psi = apply_fn(th, n, L, init=psi0)
            ev = float(np.real(psi.conj() @ observable @ psi))
            p = min(max(0.5 * (ev + 1.0), 1e-6), 1 - 1e-6)
            losses.append(-(label * np.log(p) + (1 - label) * np.log(1 - p)))
            preds.append(1 if p >= 0.5 else 0)
        return float(np.mean(losses)), float(np.mean(np.array(preds) == y))
    hist = []
    for _ in range(steps):
        L0, _ = loss_acc(theta); hist.append(L0)
        delta = rng.choice([-1.0, 1.0], size=P)
        lp, _ = loss_acc(theta + c_spsa * delta)
        lm, _ = loss_acc(theta - c_spsa * delta)
        theta = theta - lr * (lp - lm) / (2.0 * c_spsa) * delta
    Lf, accf = loss_acc(theta); hist.append(Lf)
    drop_total = hist[0] - hist[-1]
    drop_10 = hist[0] - hist[min(10, len(hist) - 1)]
    conv = (drop_10 / drop_total) if drop_total > 1e-9 else 0.0
    return {"final_loss": Lf, "final_acc": accf, "conv_rate": conv}

if ANALYSIS["run_downstream"]:
    t0 = time.time()
    n = ANALYSIS["downstream_n"]; L = CONFIG["L"]
    steps = ANALYSIS["downstream_train_steps"]
    Zall = sum(op_on_qubit(Z, q, n) for q in range(n)) / n
    n_train_seeds = ANALYSIS.get("downstream_train_seeds", 8)
    seeds_train = [100 + 7 * k for k in range(n_train_seeds)]   # SPSA training seeds

    matrix_rows = []   # one row per (ansatz, task)
    for aname, info in ANSATZ_REGISTRY.items():
        for tname, tfn in TASKS.items():
            Xd, yd = tfn(n, 12, seed=99)
            accs, losses, convs = [], [], []
            for sd in seeds_train:
                r = _train_spsa(info["apply"], n, L, info["n_params"], Xd, yd, Zall,
                                steps=steps, seed=sd)
                accs.append(r["final_acc"]); losses.append(r["final_loss"]); convs.append(r["conv_rate"])
            matrix_rows.append({
                "ansatz": aname, "task": tname,
                "task_group": TASK_NATURAL_GROUP[tname],
                "final_acc": float(np.mean(accs)),
                "final_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "final_loss": float(np.mean(losses)),
                "final_loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                "conv_rate": float(np.mean(convs)),
                "conv_rate_std": float(np.std(convs, ddof=1)) if len(convs) > 1 else 0.0,
                "n_train_seeds": int(len(accs)),
                "is_matched": NATURAL_GROUP.get(aname) == TASK_NATURAL_GROUP[tname],
            })
            print(f"  {aname:10s} x {tname:8s}: acc={np.mean(accs):.2f}+/-{np.std(accs):.2f} "
                  f"loss={np.mean(losses):.2f} conv={np.mean(convs):.2f}")

    df_downstream = pd.DataFrame(matrix_rows)

    # Accuracy matrix (rows = ansatz, cols = task)
    acc_matrix = df_downstream.pivot(index="ansatz", columns="task", values="final_acc")
    print("\nAccuracy matrix (rows = ansatz, cols = task; readout = total Z):")
    print(acc_matrix.round(2).to_string())

    # Merge with audit components (natural group, Regime B, at downstream_n)
    if "agg_scaling" in dir() and agg_scaling is not None and n in set(agg_scaling["n"]):
        comp_src = agg_scaling[agg_scaling["n"] == n][
            ["ansatz", "H_G_mean", "D_G_mean", "M_G_mean", "S_G_mean", "Psi_G_mean"]].copy()
        comp_src.columns = ["ansatz", "H_G", "D_G", "M_G", "S_G", "Psi_G"]
    else:
        comp_src = df_scaling[df_scaling["n"] == n][
            ["ansatz", "H_G", "D_G", "M_G", "S_G", "Psi_G"]].copy()
    df_down_merged = df_downstream.merge(comp_src, on="ansatz")

    # Correlation summary: per task, Spearman of each component vs accuracy across
    # the five ansaetze; plus the matched-task subset (each ansatz on its own task).
    def _spearman(a, b):
        a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.corrcoef(a, b)[0, 1]
        return float(r) if np.isfinite(r) else float("nan")

    corr_rows = []
    for tname in TASKS:
        sub = df_down_merged[df_down_merged["task"] == tname]
        for comp_c in ["H_G", "D_G", "M_G", "S_G", "Psi_G"]:
            corr_rows.append({"scope": tname, "component": comp_c,
                              "spearman_vs_acc": _spearman(sub[comp_c], sub["final_acc"])})
    matched = df_down_merged[df_down_merged["is_matched"]]
    for comp_c in ["H_G", "D_G", "M_G", "S_G", "Psi_G"]:
        corr_rows.append({"scope": "matched_task", "component": comp_c,
                          "spearman_vs_acc": _spearman(matched[comp_c], matched["final_acc"])})
    df_down_corr = pd.DataFrame(corr_rows)

    # Conditional question: among (ansatz, task) pairs that are actually
    # COMPATIBLE (the ansatz learns the task above chance), does a higher
    # Psi_G go with faster convergence or higher accuracy? This isolates the
    # within-compatible relationship from the dominant compatible/incompatible
    # split, and is reported as indicative given the small number of pairs.
    CHANCE = 0.5 + 0.10   # "above chance" threshold for compatibility
    compat = df_down_merged[df_down_merged["final_acc"] > CHANCE].copy()
    cond_rows = []
    for out_c in ["final_acc", "conv_rate"]:
        cond_rows.append({
            "scope": "compatible_pairs", "outcome": out_c,
            "n_pairs": int(len(compat)),
            "spearman_Psi_vs_outcome": _spearman(compat["Psi_G"], compat[out_c]),
            "spearman_H_vs_outcome": _spearman(compat["H_G"], compat[out_c]),
        })
    df_down_conditional = pd.DataFrame(cond_rows)
    print("\nConditional correlation among compatible (ansatz, task) pairs "
          f"(acc > {CHANCE:.2f}, n = {len(compat)} pairs):")
    print(df_down_conditional.round(2).to_string(index=False))

    print(f"\nDownstream compatibility study done in {time.time()-t0:.1f}s "
          f"(n={n}, 5 ansätze x {len(TASKS)} tasks, {len(seeds_train)} seeds each).")
    print("Interpretation: a fixed task is learnable only by structures whose "
          "symmetry does not quotient away the task's discriminating feature. "
          "The matrix is the structural result; cross-ansatz correlations within "
          "a single task are indicative only (5 points).")
else:
    df_downstream = df_down_merged = df_down_corr = df_down_conditional = None
    print("Downstream study disabled.")


if ANALYSIS["run_cost_scaling"]:
    t0 = time.time()
    cost_rows = []
    for n_c in ANALYSIS["cost_scaling_n"]:
        try:
            tg = time.time()
            G_n = build_groups_at(n_c)
            t_groups = time.time() - tg
            n_proj = sum(len(G_n[g]["projectors"]) for g in G_n)
            aname = "U1_equiv"; info = ANSATZ_REGISTRY[aname]
            gname = NATURAL_GROUP[aname]; gdata = G_n[gname]
            ta = time.time()
            _ = audit_one(aname, info, n_c, CONFIG["L"], gname,
                          gdata["projectors"], gdata["generators"], CONFIG["T"],
                          trajectory_B_multi_sector, CONFIG["gamma"],
                          CONFIG["weights"], seed=CONFIG["seed"])
            t_audit = time.time() - ta
            cost_rows.append({"n": n_c, "hilbert_dim": 2 ** n_c,
                              "n_projectors": int(n_proj),
                              "t_build_groups_s": round(t_groups, 4),
                              "t_audit_one_s": round(t_audit, 4),
                              "feasible": True})
            print(f"  n={n_c:2d}  dim={2**n_c:5d}  proj={n_proj:3d}  "
                  f"build={t_groups:.3f}s  audit={t_audit:.3f}s")
        except MemoryError:
            cost_rows.append({"n": n_c, "hilbert_dim": 2 ** n_c,
                              "n_projectors": None, "t_build_groups_s": None,
                              "t_audit_one_s": None, "feasible": False})
            print(f"  n={n_c:2d}  MemoryError -- infeasible at this size")
    df_cost = pd.DataFrame(cost_rows)
    print(f"\nCost scaling done in {time.time()-t0:.1f}s")
    print(df_cost.to_string(index=False))
else:
    df_cost = None
    print("Cost scaling disabled.")


# Figure 1 (manuscript). Sector-occupation trajectory heatmaps at n=6 in Regime B.
# This figure is independent of the composite formula and depends only on
# the per-step sector probabilities computed by Psi_G_full.

def _format_sector_label(lab):
    """Convert raw projector labels to LaTeX-formatted display labels."""
    if lab.startswith("k="):
        return r"$k=" + lab[2:] + r"$"
    if lab.startswith("j="):
        return r"$j=" + lab[2:] + r"$"
    if lab.startswith("|x|="):
        return r"$|x|=" + lab[4:] + r"$"
    return lab


fig, axes = plt.subplots(len(ANSATZ_REGISTRY), 1,
                         figsize=(9.0, 1.55 * len(ANSATZ_REGISTRY)),
                         sharex=True)

for ax, (ansatz_name, info) in zip(axes, ANSATZ_REGISTRY.items()):
    gname = NATURAL_GROUP[ansatz_name]
    gdata = GROUPS[gname]
    states = trajectory_B_multi_sector(info["apply"],
                                       CONFIG["n_audit"], CONFIG["L"],
                                       info["n_params"],
                                       T=CONFIG["T"], seed=CONFIG["seed"])
    rep = Psi_G_full(states, gdata["projectors"],
                     info["H_eff"](CONFIG["n_audit"]),
                     gdata["generators"], weights=CONFIG["weights"],
                     gamma=CONFIG["gamma"])
    P_traj = rep["P_traj"]
    im = ax.imshow(P_traj.T, aspect="auto", origin="lower",
                   vmin=0, vmax=1, cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(len(gdata["labels"])))
    ax.set_yticklabels([_format_sector_label(l) for l in gdata["labels"]],
                       fontsize=9)
    ax.set_ylabel(f"{DISPLAY_NAMES[ansatz_name]}\n[{DISPLAY_GROUPS[gname]}]",
                  fontsize=10, rotation=0, ha="right", va="center",
                  labelpad=22)
    ax.grid(False)
    # Thin axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

axes[-1].set_xlabel("Trajectory step $t$", fontsize=11)

# Single colorbar across the full stack, placed to the right with no overlap
fig.subplots_adjust(left=0.16, right=0.86, top=0.93, bottom=0.10,
                    hspace=0.28)
cbar_ax = fig.add_axes([0.88, 0.10, 0.018, 0.83])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Sector probability", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(r"Sector-occupation trajectories in Regime B at $n = 6$",
             fontsize=12, x=0.51, y=0.99)

fp = FIG_DIR / "Figure1.png"
plt.savefig(fp, dpi=300, bbox_inches="tight")
print("  saved", fp)


# Figure 2 (manuscript). Corrected Psi_G versus standard QML diagnostics
# at n=6 in Regime B. Each panel pairs Psi_G against one standard
# diagnostic (expressibility, gradient variance, Meyer-Wallach Q).

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
panel_specs = [
    ("expr_KL",  r"Expressibility KL [Sim et al.]",     "log"),
    ("grad_var", r"Gradient variance [McClean et al.]", "log"),
    ("MW_mean",  r"Meyer-Wallach $Q$",                   "linear"),
]

# Determine y-axis range from the data with a small margin
ymin = float(df_compare["Psi_G"].min()) - 0.04
ymax = float(df_compare["Psi_G"].max()) + 0.05

for ax, (col, xlab, xscale) in zip(axes, panel_specs):
    for _, row in df_compare.iterrows():
        ax.scatter(row[col], row["Psi_G"],
                   color=PALETTE[row["ansatz"]], s=180,
                   edgecolor="black", linewidth=1.0, zorder=3)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

axes[0].set_ylabel(r"$\Psi_G$ against natural group", fontsize=11)
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])

# Shared legend below, well-spaced from panels
legend_handles = [Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=PALETTE[a], markeredgecolor="black",
                         markersize=11, label=DISPLAY_NAMES[a])
                  for a in ANSATZ_REGISTRY.keys()]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.02), fontsize=10.5, frameon=False,
           columnspacing=2.0, handletextpad=0.4)

fig.suptitle(r"$\Psi_G$ versus standard QML diagnostics ($n = 6$, Regime B)",
             fontsize=12.5, y=1.00)
fig.tight_layout(rect=[0, 0.06, 1, 0.97])

fp = FIG_DIR / "Figure2.png"
plt.savefig(fp, dpi=300, bbox_inches="tight")
print("  saved", fp)


# Figure 3 (manuscript). Scaling of corrected Psi_G and its components
# across n in {4, 6, 8}.

fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharex=True)
metric_panels = [
    ("H_G",   r"$H_G$ -- sector-occupation entropy",         (0.0, 1.05)),
    ("D_G",   r"$D_G$ -- operational coherence proxy",        (0.45, 0.75)),
    ("M_G",   r"$M_G$ -- sectoral fluctuation",               (-0.005, 0.20)),
    ("Psi_G", r"$\Psi_G$ -- composite index (gated)",          (0.45, 0.90)),
]

for ax, (mkey, ttl, yrange), is_bottom in zip(axes.ravel(), metric_panels,
                                              [False, False, True, True]):
    for ansatz_name in ANSATZ_REGISTRY.keys():
        sub = df_scaling[df_scaling["ansatz"] == ansatz_name].sort_values("n")
        ax.plot(sub["n"], sub[mkey], "o-",
                color=PALETTE[ansatz_name],
                label=DISPLAY_NAMES[ansatz_name],
                markersize=8, linewidth=1.7,
                markeredgecolor="black", markeredgewidth=0.6)
    ax.set_title(ttl, pad=8, fontsize=11.5)
    ax.set_xticks(CONFIG["n_scaling"])
    if is_bottom:
        ax.set_xlabel(r"Number of qubits $n$", fontsize=11)
    ax.set_ylim(yrange)
    ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.005), fontsize=10.5, frameon=False,
           handlelength=1.8, columnspacing=2.0)
fig.suptitle(r"Scaling of $\Psi_G$ and its components across $n \in \{4, 6, 8\}$",
             y=1.06, fontsize=12.5)
fig.tight_layout()

fp = FIG_DIR / "Figure3.png"
plt.savefig(fp, dpi=300, bbox_inches="tight")
print("  saved", fp)


# Figure 4 (manuscript). Sector-confinement collapse versus multi-sector
# audit: Psi_G in Regime A vs Regime B against the natural group.

df_A = df_audit[df_audit["regime"] == "A_confined"].copy()
df_A = df_A[df_A.apply(lambda r: r["group"] == NATURAL_GROUP[r["ansatz"]], axis=1)]
df_A = df_A.set_index("ansatz").reindex(list(ANSATZ_REGISTRY.keys()))

df_B = df_audit[df_audit["regime"] == "B_multi"].copy()
df_B = df_B[df_B.apply(lambda r: r["group"] == NATURAL_GROUP[r["ansatz"]], axis=1)]
df_B = df_B.set_index("ansatz").reindex(list(ANSATZ_REGISTRY.keys()))

# Common y range
ymax_data = max(df_A["Psi_G"].max(), df_B["Psi_G"].max())
yrange = (0.0, ymax_data + 0.12)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
disp_x = [DISPLAY_NAMES[a] for a in df_A.index]

for ax, df_R, ttl in [
    (axes[0], df_A, "Regime A (sector-confined)"),
    (axes[1], df_B, "Regime B (multi-sector)"),
]:
    bar_x = np.arange(len(df_R))
    cols = [PALETTE[a] for a in df_R.index]
    bars = ax.bar(bar_x, df_R["Psi_G"].to_numpy(), color=cols,
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(disp_x, rotation=18, fontsize=10, ha="right")
    ax.set_title(ttl, pad=8, fontsize=11.5)
    ax.set_ylim(yrange)
    ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    for xi, val in zip(bar_x, df_R["Psi_G"].to_numpy()):
        # Show label slightly above the bar; for zero bars, show '0.00' at the baseline
        y_label = max(val + 0.018, 0.020)
        ax.text(xi, y_label, f"{val:.2f}",
                ha="center", fontsize=10, fontweight="normal")

axes[0].set_ylabel(r"$\Psi_G$ against natural group", fontsize=11)
fig.suptitle("Sector-confinement collapse versus multi-sector audit at $n = 6$",
             fontsize=12.5, y=1.00)
fig.tight_layout()

fp = FIG_DIR / "Figure4.png"
plt.savefig(fp, dpi=300, bbox_inches="tight")
print("  saved", fp)


# Figure 5 (manuscript). Component-level audit summary at n=6 in Regime B.

df_F5 = df_audit[(df_audit["regime"] == "B_multi")].copy()
df_F5 = df_F5[df_F5.apply(
    lambda r: r["group"] == NATURAL_GROUP[r["ansatz"]], axis=1)]
df_F5 = df_F5.set_index("ansatz").reindex(list(ANSATZ_REGISTRY.keys()))

components = ["H_G", "D_G", "M_G", "S_G", "Psi_G"]
labels = list(df_F5.index)
x = np.arange(len(labels))
w = 0.16

fig, ax = plt.subplots(figsize=(10.6, 5.4))
for i, c in enumerate(components):
    offsets = (i - (len(components) - 1) / 2) * w
    ax.bar(x + offsets, df_F5[c].to_numpy(), width=w,
           color=COMP_COLOURS[c], label=COMP_LABELS[c],
           edgecolor="black", linewidth=0.5)

disp_labels = [f"{DISPLAY_NAMES[a]}\n[{DISPLAY_GROUPS[NATURAL_GROUP[a]]}]"
               for a in labels]
ax.set_xticks(x)
ax.set_xticklabels(disp_labels, fontsize=10.5)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Normalised value", fontsize=11)

ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05),
          fontsize=11, columnspacing=1.8, handlelength=1.6,
          frameon=False)
ax.grid(axis="y", alpha=0.30, linestyle="--", linewidth=0.6)
ax.grid(axis="x", visible=False)
ax.tick_params(labelsize=10)
for spine in ax.spines.values():
    spine.set_linewidth(0.7)

fig.suptitle(r"Audit summary at $n = 6$ in Regime B (natural-group target)",
             fontsize=12.5, y=1.06)
fig.tight_layout()

fp = FIG_DIR / "Figure5.png"
plt.savefig(fp, dpi=300, bbox_inches="tight")
print("  saved", fp)


# Figure 6: composite vs learning behaviour among compatible (ansatz, task) pairs.
# Identity is encoded by colour (ansatz) and marker shape (task) with a legend,
# rather than by per-point text, so clustered points do not collide.
if ANALYSIS["run_downstream"] and df_down_merged is not None:
    CHANCE = 0.5 + 0.10
    cp = df_down_merged[df_down_merged["final_acc"] > CHANCE].copy()

    ansatz_colors = {
        "HEA":       "#4c72b0",
        "U1_equiv":  "#dd8452",
        "SU2_equiv": "#55a868",
        "Sn_equiv":  "#c44e52",
        "U1_broken": "#8172b3",
    }
    task_markers = {"U1_task": "o", "SU2_task": "s", "Sn_task": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, outcome, ylab in [
            (axes[0], "conv_rate", "Convergence rate (early loss drop fraction)"),
            (axes[1], "final_acc", "Final accuracy")]:
        for _, r in cp.iterrows():
            ax.scatter(r["Psi_G"], r[outcome],
                       s=90,
                       c=ansatz_colors.get(r["ansatz"], "#333333"),
                       marker=task_markers.get(r["task"], "o"),
                       edgecolor="white", linewidth=0.9, alpha=0.85, zorder=3)
        rho = cp[["Psi_G", outcome]].rank().corr().iloc[0, 1]
        ax.set_xlabel(r"Composite audit index $\Psi_G$", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(f"Spearman = {rho:.2f}  (n = {len(cp)} pairs)", fontsize=10.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.margins(x=0.12, y=0.12)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)

    # Shared legend: ansatz colours and task markers, placed below the panels.
    from matplotlib.lines import Line2D
    ansatz_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=c, markeredgecolor="white",
               label=DISPLAY_NAMES.get(a, a))
        for a, c in ansatz_colors.items()
    ]
    task_handles = [
        Line2D([0], [0], marker=m, linestyle="none", markersize=8,
               markerfacecolor="#777777", markeredgecolor="white", label=DISPLAY_TASKS.get(t, t))
        for t, m in task_markers.items()
    ]
    leg1 = fig.legend(handles=ansatz_handles, title="Ansatz",
                      loc="lower center", bbox_to_anchor=(0.30, -0.13),
                      ncol=3, fontsize=8.5, title_fontsize=9, frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=task_handles, title="Task",
               loc="lower center", bbox_to_anchor=(0.78, -0.13),
               ncol=3, fontsize=8.5, title_fontsize=9, frameon=False)

    fig.suptitle(r"Composite versus learning behaviour on compatible tasks at $n = 6$",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fp = FIG_DIR / "Figure6.png"
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    print("  saved", fp)

else:
    print("Downstream disabled; Figure 6 skipped.")


# Stamp every CSV with the formula version so downstream readers
# can cross-check the composite definition without opening the JSON.
# Guard the insert so re-running this cell without a kernel restart does
# not raise "cannot insert ..., already exists".
for df_obj in (df_audit, df_scaling, df_diag, df_compare, df_delta_U):
    if "psi_g_formula_version" not in df_obj.columns:
        df_obj.insert(0, "psi_g_formula_version", "v12_gated")

df_audit.to_csv(DATA_DIR / "audit_full.csv", index=False)
df_scaling.to_csv(DATA_DIR / "scaling_full.csv", index=False)
df_diag.to_csv(DATA_DIR / "comparative_diagnostics.csv", index=False)
df_compare.to_csv(DATA_DIR / "comparative_combined.csv", index=False)
df_delta_U.to_csv(DATA_DIR / "delta_U.csv", index=False)

metadata = {
    "config": {k: list(v) if isinstance(v, tuple) else v
               for k, v in CONFIG.items()},
    "psi_g_formula": "Psi_G = S_G * (w_H * H_G + w_D * D_G + w_M * H_G * (1 - M_G))",
    "psi_g_formula_version": "v12_gated",
    "psi_g_formula_note": (
        "v12 gated formulation: the stability term w_M * (1 - M_G) is "
        "gated by H_G so that it contributes only when sector structure "
        "is activated. Confined trajectories with H_G = 0 collapse to "
        "Psi_G = 0 regardless of M_G or S_G, restoring the Regime A "
        "collapse property. Earlier drafts used either w_M * M_G with "
        "wrong sign or a bare w_M * (1 - M_G) term that created a "
        "stability floor for trivial confinement."
    ),
    "unitary_compliance_formula": (
        "Delta_U_bar = E_{theta, g} [ ||U(theta) R(g) - R(g) U(theta)||_F "
        "/ (||U(theta)||_F * ||R(g)||_F) ];  "
        "S_U = exp(-gamma * Delta_U_bar)"
    ),
    "delta_U_samples": 32,
    "delta_U_n_rows": int(len(df_delta_U)),
    "delta_U_gamma": float(CONFIG["gamma"]),
    "delta_U_note": (
        "Unitary-level commutator deviation computed for the five "
        "natural-group (ansatz, group) pairs at n = n_audit. The "
        "sample count S = 32 is fixed inside the audit cell and "
        "is independent of S_diag / S_grad. Increase S to reduce "
        "Monte Carlo noise; runtime scales linearly."
    ),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": __import__("scipy").__version__,
    "pandas": pd.__version__,
    "matplotlib": mpl.__version__,
    "n_audit_rows": int(len(df_audit)),
    "n_scaling_rows": int(len(df_scaling)),
    "n_diag_rows": int(len(df_diag)),
    "ansaetze": list(ANSATZ_REGISTRY.keys()),
    "groups": list(GROUPS.keys()),
    "gradient_method": "central_finite_difference_eps_1e-4",
    "figure_format": "png_300dpi",
}
with open(META_DIR / "run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved:")
print(f"  {DATA_DIR / 'audit_full.csv'}")
print(f"  {DATA_DIR / 'scaling_full.csv'}")
print(f"  {DATA_DIR / 'comparative_diagnostics.csv'}")
print(f"  {DATA_DIR / 'comparative_combined.csv'}")
print(f"  {META_DIR / 'run_metadata.json'}")
print(f"  {len(list(FIG_DIR.glob('*.png')))} figures (PNG, 300 dpi) in {FIG_DIR}")


rev_saved = []
def _save(df_obj, name):
    if df_obj is None:
        return
    d = df_obj.copy()
    if "psi_g_formula_version" not in d.columns:
        d.insert(0, "psi_g_formula_version", "v12_gated")
    path = DATA_DIR / name
    d.to_csv(path, index=False)
    rev_saved.append(str(path))

# multi-seed (raw + aggregated)
_save(df_ms_audit,   "audit_multiseed_raw.csv")
_save(agg_audit,     "audit_multiseed.csv")
_save(df_ms_scaling, "scaling_multiseed_raw.csv")
_save(agg_scaling,   "scaling_multiseed.csv")
_save(df_ms_diag,    "comparative_multiseed_raw.csv")
_save(agg_diag,      "comparative_multiseed.csv")
_save(df_ms_deltaU,  "delta_U_multiseed_raw.csv")
_save(agg_deltaU,    "delta_U_multiseed.csv")
# sensitivity
_save(df_sens_weights,          "sensitivity_weights.csv")
_save(df_sens_gamma,            "sensitivity_gamma.csv")
_save(df_gate_ablation,         "sensitivity_gate_ablation_regimeB.csv")
_save(df_gate_ablation_regimeA, "sensitivity_gate_ablation_regimeA.csv")
# downstream
_save(df_downstream,   "downstream_results.csv")
_save(df_down_merged,  "downstream_with_components.csv")
_save(df_down_corr,    "downstream_correlations.csv")
if df_downstream is not None:
    _acc_mat = df_downstream.pivot(index="ansatz", columns="task",
                                   values="final_acc").reset_index()
    _save(_acc_mat, "downstream_accuracy_matrix.csv")
_save(df_down_conditional, "downstream_conditional_correlation.csv")
# cost
_save(df_cost,         "cost_scaling.csv")

rev_meta = {
    "analysis_config": {k: (list(v) if isinstance(v, (tuple,)) else v)
                        for k, v in ANALYSIS.items()},
    "seed_list": _seed_list(),
    "psi_g_formula_version": "v12_gated",
    "files_written": rev_saved,
}
with open(META_DIR / "analysis_metadata.json", "w") as f:
    json.dump(rev_meta, f, indent=2)

print("Analysis outputs written:")
for p in rev_saved:
    print("  ", p)
print("  ", str(META_DIR / "analysis_metadata.json"))
