#!/usr/bin/env python3
"""
Final Diagnostic Test
=====================

The shuffled control revealed density artifact, but sparse control passed.
This test resolves the contradiction.

Question: Is the sparse result genuine structure or a statistical fluke?
"""

import numpy as np
from scipy import stats
import sys


def generate_zeta_zeros(n: int = 100) -> np.ndarray:
    """First 100 Zeta zeros (verified)."""
    zeros = np.array([
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
        40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248,
        59.347044, 60.831779, 65.112544, 67.079811, 69.546402, 72.067158,
        75.704691, 77.144840, 79.337375, 82.910381, 84.735493, 87.425274,
        88.809111, 92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
        103.725538, 105.446623, 107.168611, 111.029536, 111.874659, 114.320220,
        116.226680, 118.790782, 121.370125, 122.946829, 124.256819, 127.516683,
        129.578704, 131.087688, 133.497737, 134.756509, 138.116042, 139.736208,
        141.123707, 143.111845, 146.000982, 147.422765, 150.053183, 150.925257,
        153.024693, 156.112909, 157.597591, 158.849988, 161.188964, 163.030709,
        165.537069, 167.184439, 169.094515, 169.911976, 173.411536, 174.754191,
        176.441434, 178.377407, 179.916484, 182.207078, 184.874467, 185.598783,
        187.228922, 189.416158, 192.026656, 193.079726, 195.265396, 196.876481,
        198.015309, 201.264751, 202.493594, 204.189671, 205.394698, 207.906258,
        209.576509, 211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
        220.714918, 221.430705, 224.007000, 224.983324, 227.421444, 229.337413,
        231.250188, 231.987235, 233.693404, 236.524229
    ])
    return zeros[:n]


def diagnostic_test_uniform_vs_structured(p: int = 100003, sparsity: int = 10, n_zeros: int = 100):
    """
    Test if sparse survivors have structure or are effectively uniform.

    Key insight: If survivors are uniformly spaced (arithmetic progression),
    they're guaranteed to be closer to targets than random points of same density.

    This explains both shuffled (density) and sparse (uniform structure) results.
    """
    print("="*70)
    print("DIAGNOSTIC TEST: Uniform Spacing Artifact")
    print("="*70)
    print("\nQuestion: Is sparse correlation due to uniform integer spacing?")
    print(f"Testing: p={p}, sparsity=1/{sparsity}\n")

    # K=4 survivors
    K = 4
    safe_window = (p - 1) // K
    survivors_full = np.arange(0, safe_window + 1)
    survivors_sparse = survivors_full[::sparsity]

    print(f"Full survivors: {len(survivors_full)} (uniformly spaced integers)")
    print(f"Sparse survivors: {len(survivors_sparse)} (every {sparsity}th integer)")

    # Zeta zeros
    zeros = generate_zeta_zeros(n_zeros)
    norm_zeros = (zeros % p) / p

    # Test 1: Actual sparse survivors (uniform integers)
    norm_sparse = survivors_sparse / p

    sparse_dists = []
    for z in norm_zeros:
        sparse_dists.append(np.min(np.abs(norm_sparse - z)))
    sparse_mean = np.mean(sparse_dists)

    print(f"\nActual sparse survivors mean distance: {sparse_mean:.8f}")

    # Test 2: Random sparse (same density, not uniform)
    random_sparse = np.random.uniform(0, 1, len(survivors_sparse))
    random_sparse.sort()  # Sort to be fair

    random_dists = []
    for z in norm_zeros:
        random_dists.append(np.min(np.abs(random_sparse - z)))
    random_mean = np.mean(random_dists)

    print(f"Random sparse mean distance: {random_mean:.8f}")

    # Test 3: Uniform grid (same density as sparse survivors)
    grid_sparse = np.linspace(0, 1, len(survivors_sparse))

    grid_dists = []
    for z in norm_zeros:
        grid_dists.append(np.min(np.abs(grid_sparse - z)))
    grid_mean = np.mean(grid_dists)

    print(f"Uniform grid mean distance: {grid_mean:.8f}")

    # Comparison
    print("\n" + "-"*70)
    print("COMPARISON:")
    print(f"  Sparse survivors: {sparse_mean:.8f}")
    print(f"  Random (same n):  {random_mean:.8f}")
    print(f"  Uniform grid:     {grid_mean:.8f}")

    # Key test: Is sparse survivors closer to uniform grid than to random?
    uniform_like = abs(sparse_mean - grid_mean) < abs(sparse_mean - random_mean)

    print(f"\n  Sparse is closer to: {'UNIFORM GRID' if uniform_like else 'RANDOM'}")

    # Statistical test: sparse vs uniform grid
    # If they're statistically indistinguishable, sparse "correlation" is just uniform spacing
    t_stat, p_value = stats.ttest_ind(sparse_dists, grid_dists)

    print(f"\n  t-test (sparse vs grid): p={p_value:.6f}")

    if p_value > 0.05:
        print("  → NOT significantly different (uniform spacing artifact)")
    else:
        print("  → Significantly different (genuine structure)")

    # Another test: Kolmogorov-Smirnov
    ks_stat, ks_p = stats.ks_2samp(sparse_dists, grid_dists)
    print(f"  KS-test (sparse vs grid): p={ks_p:.6f}")

    if ks_p > 0.05:
        print("  → Distributions match (uniform spacing artifact)")
    else:
        print("  → Distributions differ (genuine structure)")

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)

    if p_value > 0.05 and ks_p > 0.05:
        print("\n✗ SPARSE RESULT IS UNIFORM SPACING ARTIFACT")
        print("\nExplanation:")
        print("  Survivors are uniformly spaced integers: {0, 1, 2, 3, ...}")
        print("  This creates a regular grid in [0, 1] when normalized")
        print("  Regular grids are always closer to targets than random points")
        print("  This is guaranteed by geometry, not Zeta structure")
        print("\n  The 'correlation' is:")
        print("    1. High density (25% of residues) - density artifact")
        print("    2. Uniform spacing (arithmetic progression) - grid artifact")
        print("\n  Both artifacts, no genuine Zeta coupling.")

    elif uniform_like:
        print("\n⚠ SPARSE RESULT LARGELY ARTIFACTUAL")
        print("  Mostly due to uniform grid structure")
        print("  May have small genuine component")

    else:
        print("\n✓ SPARSE RESULT SHOWS GENUINE STRUCTURE")
        print("  Not explained by uniform spacing alone")
        print("  Possible genuine Zeta coupling")

    return {
        'sparse_mean': sparse_mean,
        'random_mean': random_mean,
        'grid_mean': grid_mean,
        'uniform_like': uniform_like,
        'ttest_p': p_value,
        'ks_p': ks_p,
        'artifact_confirmed': p_value > 0.05 and ks_p > 0.05
    }


if __name__ == "__main__":
    print("\nRunning final diagnostic to resolve sparse control contradiction...\n")
    result = diagnostic_test_uniform_vs_structured(p=100003, sparsity=10, n_zeros=100)

    print("\n" + "="*70)
    print("FINAL ANSWER")
    print("="*70)

    if result['artifact_confirmed']:
        print("\nThe observed 'Zeta correlation' is FULLY EXPLAINED by:")
        print("  1. DENSITY: 25% of residues are survivors")
        print("  2. UNIFORMITY: Survivors are regular integer grid")
        print("\nNo genuine connection to Riemann Zeta zeros detected.")
        print("\nVerdict: ZETA COUPLING FALSIFIED")
    else:
        print("\nMixed evidence. Some genuine structure may exist")
        print("beyond density and uniformity artifacts.")
        print("\nVerdict: INCONCLUSIVE - requires theoretical analysis")
