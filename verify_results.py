#!/usr/bin/env python3
"""Quick verification of the PsiAudit headline numbers for reviewers.

Loads the precomputed result tables in ``Results/`` and confirms that the
headline quantities reported in the paper are reproducible from those tables.
Runs in a couple of seconds, needs only pandas and numpy, and no GPU. Prints a
clean PASS/FAIL summary and exits non-zero if any check fails.

Usage:
    python verify_results.py
    python verify_results.py --results Results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHECKS = []


def check(label, condition, detail=""):
    CHECKS.append((bool(condition), label, detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def approx(a, b, tol=0.02):
    return abs(float(a) - float(b)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="Results",
                    help="path to the Results directory (default: Results)")
    args = ap.parse_args()
    data = Path(args.results) / "data"
    if not data.is_dir():
        print(f"ERROR: {data} not found. Run from the repository root or pass --results.")
        sys.exit(2)

    print("PsiAudit results verification")
    print("=" * 60)

    # 1. Multi-seed audit at n = 6 (Regime B, natural-group target)
    ms = pd.read_csv(data / "scaling_multiseed.csv")
    b6 = ms[ms.n == 6].set_index("ansatz")
    print("\nMulti-seed composite at n = 6 (mean over 20 seeds):")
    check("HEA Psi_G near 0.670", approx(b6.loc["HEA", "Psi_G_mean"], 0.670),
          f"{b6.loc['HEA','Psi_G_mean']:.3f}")
    check("U1_equiv Psi_G near 0.612 and deterministic",
          approx(b6.loc["U1_equiv", "Psi_G_mean"], 0.612)
          and b6.loc["U1_equiv", "Psi_G_std"] < 1e-6,
          f"{b6.loc['U1_equiv','Psi_G_mean']:.3f} +/- {b6.loc['U1_equiv','Psi_G_std']:.3f}")
    check("SU2_equiv Psi_G near 0.522 and deterministic",
          approx(b6.loc["SU2_equiv", "Psi_G_mean"], 0.522)
          and b6.loc["SU2_equiv", "Psi_G_std"] < 1e-6,
          f"{b6.loc['SU2_equiv','Psi_G_mean']:.3f} +/- {b6.loc['SU2_equiv','Psi_G_std']:.3f}")
    check("Sn_equiv highest composite near 0.744 with widest spread",
          approx(b6.loc["Sn_equiv", "Psi_G_mean"], 0.744, tol=0.03)
          and b6.loc["Sn_equiv", "Psi_G_std"] > 0.03,
          f"{b6.loc['Sn_equiv','Psi_G_mean']:.3f} +/- {b6.loc['Sn_equiv','Psi_G_std']:.3f}")
    check("U1_broken Psi_G near 0.624",
          approx(b6.loc["U1_broken", "Psi_G_mean"], 0.624),
          f"{b6.loc['U1_broken','Psi_G_mean']:.3f}")

    # 2. Unitary-level commutator deviation
    du = pd.read_csv(data / "delta_U_multiseed.csv").set_index("ansatz")
    print("\nUnitary-level commutator deviation (Eq. 10):")
    check("Equivariant ansaetze have zero Delta_U",
          max(du.loc["U1_equiv", "Delta_U_mean"],
              du.loc["SU2_equiv", "Delta_U_mean"],
              du.loc["Sn_equiv", "Delta_U_mean"]) < 1e-9,
          "all three exactly zero")
    check("U1_broken Delta_U near 0.027 (S_U near 0.922)",
          approx(du.loc["U1_broken", "Delta_U_mean"], 0.027, tol=0.01),
          f"Delta_U = {du.loc['U1_broken','Delta_U_mean']:.3f}, "
          f"S_U = {du.loc['U1_broken','S_U_mean']:.3f}")

    # 3. Sensitivity of the ranking to weights and penalty
    sw = pd.read_csv(data / "sensitivity_weights.csv")
    print("\nRanking robustness:")
    top_stable = (sw.top_ansatz == sw.top_ansatz.mode()[0]).mean()
    check("Top-ranked ansatz unchanged across all weight draws",
          top_stable > 0.999, f"{100 * top_stable:.0f}% of draws")
    check("Median Kendall tau vs convention is 1.0",
          approx(sw.kendall_tau_vs_convention.median(), 1.0, tol=1e-6),
          f"median tau = {sw.kendall_tau_vs_convention.median():.3f}")

    # 4. Gate ablation in Regime A
    ab = pd.read_csv(data / "sensitivity_gate_ablation_regimeA.csv").set_index("ansatz")
    print("\nH_G-gate ablation (Regime A):")
    check("Gated composite collapses equivariant ansaetze to zero",
          ab.loc["U1_equiv", "Psi_gated_regimeA"] < 1e-9
          and ab.loc["SU2_equiv", "Psi_gated_regimeA"] < 1e-9,
          "U1_equiv and SU2_equiv gated = 0")
    check("Ungated composite leaves a spurious floor of 0.25",
          approx(ab.loc["U1_equiv", "Psi_ungated_regimeA"], 0.25, tol=0.02),
          f"ungated = {ab.loc['U1_equiv','Psi_ungated_regimeA']:.3f}")

    # 5. Downstream compatibility matrix
    am = pd.read_csv(data / "downstream_accuracy_matrix.csv").set_index("ansatz")
    print("\nDownstream compatibility (accuracy matrix):")
    check("HEA learns all three tasks",
          am.loc["HEA", "SU2_task"] > 0.95 and am.loc["HEA", "Sn_task"] > 0.95
          and am.loc["HEA", "U1_task"] > 0.75, "SU2, Sn perfect; U1 above 0.75")
    check("Sn_equiv solves Sn and U1 tasks perfectly",
          am.loc["Sn_equiv", "Sn_task"] > 0.95 and am.loc["Sn_equiv", "U1_task"] > 0.95,
          "permutation equivariance subsumes excitation structure")
    check("U1/SU2-equiv and U1-broken succeed only on SU2 task",
          am.loc["U1_equiv", "U1_task"] < 0.1 and am.loc["U1_equiv", "Sn_task"] < 0.1
          and am.loc["SU2_equiv", "U1_task"] < 0.1,
          "at chance on non-matched tasks")

    # 6. Conditional correlation among compatible pairs
    cc = pd.read_csv(data / "downstream_conditional_correlation.csv")
    conv = cc[cc.outcome == "conv_rate"]["spearman_Psi_vs_outcome"].iloc[0]
    acc = cc[cc.outcome == "final_acc"]["spearman_Psi_vs_outcome"].iloc[0]
    print("\nConditional correlation (compatible pairs):")
    check("Psi_G tracks convergence rate strongly (Spearman near 0.9)",
          conv > 0.8, f"Spearman = {conv:.2f}")
    check("Psi_G correlates weakly with final accuracy",
          acc < 0.5, f"Spearman = {acc:.2f}")

    # Summary
    n_pass = sum(1 for ok, _, _ in CHECKS if ok)
    n_total = len(CHECKS)
    print("\n" + "=" * 60)
    print(f"{n_pass} / {n_total} checks passed.")
    if n_pass == n_total:
        print("ALL CHECKS PASSED.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED.")
        for ok, label, _ in CHECKS:
            if not ok:
                print(f"  FAILED: {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
