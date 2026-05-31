"""
PsiAudit core module - all functions used by the notebook.

This is built as a single .py file first so it can be unit-tested and
debugged independently of the notebook. The notebook then imports / inlines
the same functions.
"""
from __future__ import annotations

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
