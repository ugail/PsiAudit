# PsiAudit: Auditing Symmetry-Organised Complexity in Quantum Neural Networks

*PsiAudit: A Reproducible Toolkit for Symmetry-Organised Complexity in Equivariant Quantum Neural Networks*, by H. Ugail and N. Howard.

Parameterised quantum circuits are usually evaluated with diagnostics for expressibility, gradient behaviour, and entanglement. These are useful, but they say nothing about whether a circuit respects the symmetry structure it was designed around, which matters most in equivariant quantum machine learning where symmetry is part of the model. **PsiAudit is an open-source toolkit that audits symmetry-organised behaviour in quantum neural network ansaetze before training.** For a given ansatz, target symmetry, and state trajectory it reports four normalised components, namely sector-occupation entropy, cross-sector coherence, sectoral fluctuation, and generator-sum compliance, and combines them into a configurable composite index that is read as a dashboard rather than as a measure of downstream accuracy. The toolkit supports unitary phase symmetry, spin symmetry, and permutation symmetry, with the permutation audit performed at the level of Hamming-weight orbits rather than full symmetric-group irreducible representations.

<img width="2660" height="2267" alt="Figure1" src="https://github.com/user-attachments/assets/9752cb74-b006-4afb-89af-8c42185e0524" />


## What does this measure?

The headline empirical findings, reproduced across twenty random seeds and system sizes from four to eight qubits, are:

1. **The audit separates ansaetze that the standard diagnostics cluster together.** The U(1)-equivariant and U(1)-broken ansaetze sit almost on top of each other under expressibility, gradient variance, and Meyer-Wallach Q, yet PsiAudit separates them cleanly through the sectoral fluctuation and the generator-sum compliance.

2. **A unitary-level compliance check sharpens the separation.** The parameter-averaged commutator deviation is exactly zero for the three equivariant ansaetze and 0.027 for the U(1)-broken near-control, a sharper reading than the generator-sum proxy provides on its own.

3. **The composite flags inactive ansaetze honestly.** An exactly equivariant ansatz scores zero when its input stays confined to one symmetry sector, and recovers a non-trivial score once the input activates several sectors, so the audit assesses the ansatz and the input regime together. An H_G-gate ablation shows this collapse is a designed property, not an accident of the inputs.

4. **The ranking is robust to the composite weights and penalty.** Across two hundred random weight vectors and a grid of penalty values, the highest-scoring ansatz is unchanged in every draw and the median Kendall tau against the convention ordering is 1.0.

5. **Downstream task reach follows symmetry alignment.** On three fixed symmetry-matched classification tasks, each architecture performs well precisely on tasks whose structure its symmetry preserves. Among compatible pairs, the composite tracks convergence rate strongly (Spearman 0.89) while its correlation with final accuracy is weak (0.26).

## Contents

- **`run_pipeline_full.ipynb`** — the full reproduction notebook. Runs the self-test battery, the two-regime audit, the scaling study, the comparative diagnostics, the multi-seed replication with confidence intervals, the sensitivity sweep and gate ablation, the downstream compatibility study, the audit-cost measurement, and the six manuscript figures. Autodetects whether it is running in Google Colab or locally.

- **`run_experiments.py`** — the same full pipeline as a single command-line script. Runs at production fidelity and writes every CSV and figure to `./results`. Pure NumPy and SciPy, CPU only.

- **`verify_results.py`** — a quick verification script for reviewers. Loads the precomputed tables in `Results/` and confirms that every headline number in the paper is reproducible from those tables. Runs in a couple of seconds, needs no GPU, and prints a clean PASS/FAIL summary.

- **`psiaudit/`** — the Python package. Contains the audit primitives, the four components, the composite, and the standard diagnostics (`core.py`), the reference ansatz library and effective generator-sum Hamiltonians (`ansatze.py`), the two trajectory builders (`trajectories.py`), and the group-projector construction with the single-audit driver (`audit.py`).

- **`scripts/quick_audit.py`** — a minimal worked example showing how to audit a built-in ansatz and how to drop in your own circuit.

- **`tests/`** — a pytest suite verifying the projector constructions, the compliance defect, the Meyer-Wallach calibration, the representation-theoretic placement of symmetric Dicke states, the multi-sector activation of Regime B, the gated-composite ordering, and the Regime A collapse.

- **`Results/`** — the precomputed result tables and figures for the paper, sufficient to verify every reported number without re-running the pipeline.

- **`requirements.txt`** — full dependencies (NumPy, SciPy, pandas, matplotlib). **`requirements-verify.txt`** — minimal dependencies for `verify_results.py` only (NumPy, pandas).

## Quick start

The fastest way to check the headline numbers is:

```
git clone https://github.com/ugail/PsiAudit.git
cd PsiAudit
pip install -r requirements-verify.txt
python verify_results.py
```

The script loads the precomputed CSVs in `Results/`, recomputes every quantity claimed in the manuscript, and prints a PASS/FAIL summary. A full pass takes a couple of seconds on any laptop.

To audit an ansatz interactively:

```
pip install -e .
python scripts/quick_audit.py
```

## Reproducing the full pipeline

Re-running the full pipeline requires Python 3.10 or later. No GPU is needed; the toolkit is pure NumPy and SciPy and runs on a standard CPU. Install the full requirements first:

```
pip install -r requirements.txt
pip install -e .
```

To reproduce all reported results from a clean run, use the script:

```
python run_experiments.py
```

It writes every CSV and figure to `./results` by default. Set `PSIAUDIT_OUT` to change the output directory. A full production run takes a few minutes on a modern CPU.

The same pipeline is available as a notebook, which autodetects whether it is running in Google Colab or locally:

```
jupyter notebook run_pipeline_full.ipynb
```

In Colab, set the environment variable `PSIAUDIT_USE_DRIVE=1` to mount and write to Google Drive; otherwise both the script and the notebook write locally.

## Tests

To run the structural invariant tests:

```
pip install pytest
pytest tests
```

The tests verify projector idempotence and completeness, that the equivariant ansatz Hamiltonians commute with their target generators while the broken ansatz is flagged, the Meyer-Wallach calibration on product and GHZ states, the placement of symmetric Dicke states in the top spin sector, the multi-sector support of Regime B, the symmetry-correct ordering of the gated composite, and the Regime A collapse of an equivariant ansatz.

## Audit components

For a target group `G`, a chosen state trajectory, and an ansatz, PsiAudit reports:

- **`H_G`** — sector-occupation entropy, the normalised Shannon entropy of the time-averaged sector-probability distribution.
- **`D_G`** — cross-sector coherence, the mean of an inter-sector and a within-multiplicity-space coherence proxy.
- **`M_G`** — sectoral fluctuation, the variation of the inverse participation ratio across the trajectory.
- **`S_G`** — generator-sum compliance, `exp(-gamma * Delta_G)` where `Delta_G` is the normalised commutator defect of the effective generator-sum Hamiltonian against the target generators.

These combine into the configurable composite `Psi_G = S_G * (w_H * H_G + w_D * D_G + w_M * H_G * (1 - M_G))`, with default weights `(0.40, 0.35, 0.25)` and penalty `gamma = 3`. The stability term is gated by `H_G` so that a sector-confined trajectory is correctly reported as carrying no activated structure. A complementary unitary-level commutator deviation is also computed as the more stringent compliance check.

## Who is this for?

- **Researchers in quantum machine learning** who want a pre-deployment screen for whether an ansatz organises its trajectory according to a chosen symmetry, complementing the usual expressibility, trainability, and entanglement diagnostics.
- **Researchers in equivariant model design** who want a worked, reproducible example of symmetry-aware ansatz auditing with explicit invariant checks and multi-seed reporting.
- **Reviewers** who want to verify every headline number from precomputed tables in a couple of seconds, without re-running the pipeline.

## Citation

If you use this toolkit or the precomputed result tables, please cite the toolkit paper and its companion theory study:

> H. Ugail and N. Howard. *PsiAudit: A Reproducible Toolkit for Symmetry-Organised Complexity in Equivariant Quantum Neural Networks*. Under review.

> H. Ugail and N. Howard. *Symmetry-Organised Complexity in Quantum Neural Networks*. Symmetry, 2026, 18(6), 912. doi:10.3390/sym18060912.

## License

Released under the MIT License. See `LICENSE` for the full text.
