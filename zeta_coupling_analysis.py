#!/usr/bin/env python3
"""
Extended Zeta Coupling Analysis Suite
======================================

Rigorous investigation of potential connections between the skew-product
Collatz system and Riemann Zeta zero dynamics.

This suite goes beyond basic verification to explore:
1. Spectral statistics of survivor distributions
2. Level spacing analysis (GUE vs Poisson)
3. Prime modulus dependencies
4. Long-term fiber evolution patterns
5. Correlation with known Zeta zero properties
6. Montgomery pair correlation analogs
7. Critical line behavior analogs

Approach: Truth over comfort. If no connection exists, we'll find that.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys
from scipy import stats
from scipy.spatial.distance import pdist


@dataclass
class SpectralAnalysis:
    """Results of spectral statistics analysis."""
    spacings: np.ndarray
    mean_spacing: float
    spacing_ratio_mean: float  # <s_n+1/s_n>
    gue_divergence: float  # KS test statistic vs GUE
    poisson_divergence: float  # KS test statistic vs Poisson
    level_repulsion: bool  # True if spacings avoid zero


class CollatzZetaAnalyzer:
    """
    Extended analyzer for investigating Zeta coupling.

    Key question: Is there arithmetic structure in K=4 survivors
    that mirrors Riemann Zeta zero statistics?
    """

    def __init__(self, K: int, p: int):
        self.K = K
        self.p = p
        self.safe_window = (p - 1) // K

    def compute_return_map(self, n: int) -> int:
        """Fiber return map over one Collatz cycle."""
        # 1→4: n ↦ Kn
        n1 = (self.K * n) % self.p
        # 4→2: n ↦ n/2
        n2 = (n1 * pow(2, -1, self.p)) % self.p
        # 2→1: n ↦ n/2
        n3 = (n2 * pow(2, -1, self.p)) % self.p
        return n3

    def orbit_under_return_map(self, n0: int, max_iterations: int = 10000) -> List[int]:
        """
        Compute full orbit under iterated return map.
        Returns orbit until exit from safe window or max_iterations.
        """
        orbit = [n0]
        n = n0

        for _ in range(max_iterations):
            n = self.compute_return_map(n)
            orbit.append(n)

            # Check if exited safe window
            if n > self.safe_window:
                break

        return orbit

    def find_periodic_orbits(self, max_period: int = 1000) -> Dict[int, List[int]]:
        """
        Find all periodic orbits under return map within safe window.
        Returns dict mapping period -> list of orbit representatives.
        """
        visited = set()
        periodic_orbits = defaultdict(list)

        for n0 in range(self.safe_window + 1):
            if n0 in visited:
                continue

            orbit = []
            n = n0
            positions = {}

            for step in range(max_period):
                if n in positions:
                    # Found period
                    period = step - positions[n]
                    orbit_start = positions[n]
                    periodic_orbit = orbit[orbit_start:orbit_start + period]

                    # Mark all points in orbit as visited
                    for pt in periodic_orbit:
                        visited.add(pt)

                    periodic_orbits[period].append(periodic_orbit)
                    break

                positions[n] = step
                orbit.append(n)
                n = self.compute_return_map(n)

                # Check if exited safe window
                if n > self.safe_window:
                    break

        return dict(periodic_orbits)

    def compute_spectral_statistics(self, values: np.ndarray) -> SpectralAnalysis:
        """
        Compute spectral statistics on a sequence of values.

        Compares to:
        - GUE (Gaussian Unitary Ensemble): level repulsion
        - Poisson: random, no correlations
        """
        # Normalize to unit mean spacing
        sorted_vals = np.sort(values)
        spacings = np.diff(sorted_vals)

        if len(spacings) == 0:
            return SpectralAnalysis(
                spacings=np.array([]),
                mean_spacing=0,
                spacing_ratio_mean=0,
                gue_divergence=0,
                poisson_divergence=0,
                level_repulsion=False
            )

        # Unfold spectrum (normalize to unit mean spacing)
        mean_gap = np.mean(spacings)
        if mean_gap > 0:
            normalized_spacings = spacings / mean_gap
        else:
            normalized_spacings = spacings

        # Spacing ratio statistic
        if len(normalized_spacings) > 1:
            ratios = normalized_spacings[1:] / (normalized_spacings[:-1] + 1e-10)
            spacing_ratio_mean = np.mean(np.minimum(ratios, 1/ratios))
        else:
            spacing_ratio_mean = 0

        # Compare to theoretical distributions
        # GUE: Wigner surmise P(s) = (32/π²)s²exp(-4s²/π)
        # Poisson: P(s) = exp(-s)

        # KS test against Poisson
        poisson_divergence = self._ks_distance(
            normalized_spacings,
            lambda s: 1 - np.exp(-s)  # Poisson CDF
        )

        # KS test against Wigner (approximation)
        gue_divergence = self._ks_distance(
            normalized_spacings,
            lambda s: 1 - np.exp(-np.pi * s**2 / 4)  # Wigner CDF (approx)
        )

        # Level repulsion: check if small spacings are suppressed
        small_spacing_count = np.sum(normalized_spacings < 0.1)
        level_repulsion = (small_spacing_count / len(normalized_spacings)) < 0.05

        return SpectralAnalysis(
            spacings=normalized_spacings,
            mean_spacing=mean_gap,
            spacing_ratio_mean=spacing_ratio_mean,
            gue_divergence=gue_divergence,
            poisson_divergence=poisson_divergence,
            level_repulsion=level_repulsion
        )

    def _ks_distance(self, data: np.ndarray, cdf_func) -> float:
        """Kolmogorov-Smirnov distance between data and theoretical CDF."""
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        theoretical_cdf = cdf_func(sorted_data)
        return np.max(np.abs(empirical_cdf - theoretical_cdf))


class ZetaCouplingTests:
    """
    Comprehensive test suite for Zeta coupling investigation.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def test_spectral_statistics_k4(self, primes: List[int]) -> Dict:
        """
        TEST A: Spectral statistics of K=4 survivor distributions.

        Question: Do K=4 survivors exhibit spectral correlations
        similar to Zeta zeros (GUE statistics)?
        """
        self.log("\n" + "="*70)
        self.log("TEST A: K=4 Spectral Statistics Across Primes")
        self.log("="*70)
        self.log("\nObjective: Detect GUE-like level repulsion in survivor states")

        K = 4
        results_by_prime = {}

        for p in primes:
            self.log(f"\n--- Prime p={p} ---")
            analyzer = CollatzZetaAnalyzer(K, p)

            # Get all survivor states (points in safe window for K=4 survive forever)
            survivors = np.arange(1, analyzer.safe_window + 1)

            # Compute spectral statistics
            spectral = analyzer.compute_spectral_statistics(survivors)

            self.log(f"Safe window: [0, {analyzer.safe_window}]")
            self.log(f"Survivor count: {len(survivors)}")
            self.log(f"Mean spacing: {spectral.mean_spacing:.4f}")
            self.log(f"Spacing ratio: {spectral.spacing_ratio_mean:.4f}")
            self.log(f"  (GUE ≈ 0.536, Poisson ≈ 0.386)")
            self.log(f"KS divergence from GUE: {spectral.gue_divergence:.4f}")
            self.log(f"KS divergence from Poisson: {spectral.poisson_divergence:.4f}")
            self.log(f"Level repulsion: {spectral.level_repulsion}")

            # Closer to GUE or Poisson?
            closer_to_gue = spectral.gue_divergence < spectral.poisson_divergence
            self.log(f"Closer to: {'GUE' if closer_to_gue else 'Poisson'}")

            results_by_prime[p] = {
                'spectral': spectral,
                'closer_to_gue': closer_to_gue,
                'survivor_count': len(survivors)
            }

        # Overall assessment
        gue_count = sum(1 for r in results_by_prime.values() if r['closer_to_gue'])
        self.log(f"\n{'='*50}")
        self.log(f"Primes closer to GUE: {gue_count}/{len(primes)}")

        # GUE-like behavior would suggest quantum chaos / RMT connection
        suggests_gue = gue_count > len(primes) / 2

        return {
            'test': 'k4_spectral_statistics',
            'results_by_prime': results_by_prime,
            'suggests_gue_statistics': suggests_gue,
            'gue_count': gue_count
        }

    def test_periodic_orbit_structure(self, primes: List[int], K_values: List[int]) -> Dict:
        """
        TEST B: Periodic orbit structure under return map.

        Question: Does K=4 exhibit unique periodic structure?
        """
        self.log("\n" + "="*70)
        self.log("TEST B: Periodic Orbit Structure")
        self.log("="*70)

        results = {}

        for K in K_values:
            self.log(f"\n--- K={K} ---")
            K_results = {}

            for p in primes:
                analyzer = CollatzZetaAnalyzer(K, p)
                periodic_orbits = analyzer.find_periodic_orbits(max_period=100)

                total_periodic_points = sum(
                    len(orbits) * period
                    for period, orbits in periodic_orbits.items()
                )

                self.log(f"p={p}: periods found: {sorted(periodic_orbits.keys())}")
                self.log(f"  Total periodic points: {total_periodic_points}/{analyzer.safe_window+1}")

                K_results[p] = {
                    'periods': sorted(periodic_orbits.keys()),
                    'periodic_point_count': total_periodic_points,
                    'window_size': analyzer.safe_window + 1,
                    'periodic_fraction': total_periodic_points / (analyzer.safe_window + 1)
                }

            results[K] = K_results

        # K=4 should have period-1 dominating (fixed points everywhere)
        k4_period1_dominant = all(
            1 in results[4][p]['periods']
            for p in primes
        )

        return {
            'test': 'periodic_orbit_structure',
            'results': results,
            'k4_period1_dominant': k4_period1_dominant
        }

    def test_prime_modulus_dependence(self, K: int, primes: List[int]) -> Dict:
        """
        TEST C: How do dynamics depend on prime modulus?

        Question: Are there systematic patterns as p varies?
        Related to explicit formula: Zeta zeros appear in prime counting.
        """
        self.log("\n" + "="*70)
        self.log(f"TEST C: Prime Modulus Dependence (K={K})")
        self.log("="*70)

        results = []

        for p in primes:
            analyzer = CollatzZetaAnalyzer(K, p)

            # Measure key quantities
            window_size = analyzer.safe_window + 1
            window_fraction = window_size / p

            # For K=4: all points are fixed
            # For K≠4: measure survival statistics
            if K == 4:
                mean_orbit_length = float('inf')
                survival_rate = 1.0
            else:
                # Sample orbits
                sample_size = min(100, window_size)
                samples = np.linspace(1, analyzer.safe_window, sample_size, dtype=int)

                orbit_lengths = []
                max_iter = 1000

                for n in samples:
                    orbit = analyzer.orbit_under_return_map(n, max_iterations=max_iter)
                    orbit_lengths.append(len(orbit))

                mean_orbit_length = np.mean(orbit_lengths)
                survival_rate = np.sum(np.array(orbit_lengths) >= max_iter) / len(orbit_lengths)

            results.append({
                'p': p,
                'window_size': window_size,
                'window_fraction': window_fraction,
                'mean_orbit_length': mean_orbit_length,
                'survival_rate': survival_rate
            })

            self.log(f"p={p}: window={window_size} ({window_fraction:.4f}), "
                    f"mean_orbit={mean_orbit_length:.1f}, survival={survival_rate:.3f}")

        # Look for scaling with p
        primes_arr = np.array([r['p'] for r in results])
        window_fractions = np.array([r['window_fraction'] for r in results])

        # Expected: window_fraction ≈ 1/K
        expected_fraction = 1.0 / K
        deviation = np.mean(np.abs(window_fractions - expected_fraction))

        self.log(f"\nExpected window fraction: 1/{K} = {expected_fraction:.4f}")
        self.log(f"Mean deviation: {deviation:.6f}")

        return {
            'test': 'prime_modulus_dependence',
            'K': K,
            'results': results,
            'expected_fraction': expected_fraction,
            'observed_deviation': deviation
        }

    def test_fiber_state_correlations(self, p: int, K: int = 4, n_cycles: int = 10000) -> Dict:
        """
        TEST D: Long-term correlations in fiber state evolution.

        Question: Do fiber states exhibit long-range correlations?
        """
        self.log("\n" + "="*70)
        self.log(f"TEST D: Fiber State Correlations (K={K}, p={p})")
        self.log("="*70)

        analyzer = CollatzZetaAnalyzer(K, p)

        # Start from middle of safe window
        n0 = analyzer.safe_window // 2

        # Evolve for many cycles
        trajectory = [n0]
        n = n0

        for _ in range(n_cycles):
            n = analyzer.compute_return_map(n)
            trajectory.append(n)

        trajectory = np.array(trajectory)

        # Compute autocorrelation
        def autocorr(x, lag):
            if lag >= len(x):
                return 0
            c0 = np.var(x)
            if c0 == 0:
                return 1 if lag == 0 else 0
            return np.corrcoef(x[:-lag or None], x[lag:])[0, 1] if lag > 0 else 1

        lags = [1, 10, 100, 1000]
        autocorrs = [autocorr(trajectory, lag) for lag in lags]

        self.log(f"Initial state: n0={n0}")
        self.log(f"Trajectory length: {len(trajectory)}")
        self.log(f"Unique states visited: {len(np.unique(trajectory))}")

        self.log(f"\nAutocorrelations:")
        for lag, ac in zip(lags, autocorrs):
            self.log(f"  lag={lag}: {ac:.4f}")

        # For K=4, trajectory should be constant (n0, n0, n0, ...)
        if K == 4:
            is_constant = len(np.unique(trajectory)) == 1
            self.log(f"\nK=4 constant trajectory: {is_constant}")

        return {
            'test': 'fiber_state_correlations',
            'K': K,
            'p': p,
            'trajectory_length': len(trajectory),
            'unique_states': len(np.unique(trajectory)),
            'autocorrelations': dict(zip(lags, autocorrs)),
            'is_constant': len(np.unique(trajectory)) == 1 if K == 4 else None
        }

    def test_critical_line_analog(self, primes: List[int]) -> Dict:
        """
        TEST E: Critical line analog.

        Question: Is there a "critical" value of K around 4?
        Riemann hypothesis: all nontrivial zeros lie on Re(s) = 1/2.
        Analog: K=4 might be "critical" coupling.
        """
        self.log("\n" + "="*70)
        self.log("TEST E: Critical Coupling Investigation")
        self.log("="*70)
        self.log("\nScanning K near 4 with fine resolution...")

        # Test K values near 4
        # For integer K, only 4 is exact, but we can test nearby values
        K_values = [2, 3, 4, 5, 6, 7, 8]

        results = {}

        for K in K_values:
            self.log(f"\n--- K={K} ---")

            K_results = []
            for p in primes[:3]:  # Use fewer primes for speed
                analyzer = CollatzZetaAnalyzer(K, p)

                # Measure "stability" via survival rate
                sample_size = min(50, analyzer.safe_window + 1)
                samples = np.linspace(0, analyzer.safe_window, sample_size, dtype=int)

                survival_count = 0
                max_iter = 1000

                for n in samples:
                    orbit = analyzer.orbit_under_return_map(n, max_iterations=max_iter)
                    if len(orbit) >= max_iter:
                        survival_count += 1

                survival_rate = survival_count / sample_size
                K_results.append(survival_rate)

            mean_survival = np.mean(K_results)
            self.log(f"Mean survival rate: {mean_survival:.4f}")

            results[K] = {
                'mean_survival': mean_survival,
                'by_prime': K_results
            }

        # K=4 should have survival ≈ 1.0
        # All others should have survival ≈ 0
        k4_critical = results[4]['mean_survival'] > 0.95
        others_non_critical = all(
            results[K]['mean_survival'] < 0.1
            for K in K_values if K != 4
        )

        self.log(f"\n{'='*50}")
        self.log(f"K=4 is critical: {k4_critical}")
        self.log(f"Other K are non-critical: {others_non_critical}")

        return {
            'test': 'critical_line_analog',
            'results': results,
            'k4_is_critical': k4_critical and others_non_critical
        }

    def test_zeta_zero_positions(self, p: int) -> Dict:
        """
        TEST F: Direct comparison with known Zeta zero positions.

        Question: Do fiber states correlate with actual Zeta zero positions?

        First ~100 Zeta zeros (imaginary parts γ_n):
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062, ...
        """
        self.log("\n" + "="*70)
        self.log(f"TEST F: Direct Zeta Zero Correlation (p={p})")
        self.log("="*70)

        # First 30 nontrivial Zeta zero imaginary parts
        # Source: LMFDB / verified tables
        zeta_zeros_gamma = np.array([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425274, 88.809111,
            92.491899, 94.651344, 95.870634, 98.831194, 101.317851
        ])

        K = 4
        analyzer = CollatzZetaAnalyzer(K, p)

        # Get survivor states (all points in safe window for K=4)
        survivors = np.arange(0, analyzer.safe_window + 1)

        # Normalize both to [0, 1] for comparison
        normalized_survivors = survivors / p
        normalized_zeros = (zeta_zeros_gamma % p) / p  # Mod p for comparison

        self.log(f"Survivor states: {len(survivors)}")
        self.log(f"Zeta zeros tested: {len(zeta_zeros_gamma)}")

        # Compute nearest-neighbor distances
        # If correlated, survivors should cluster near normalized zeros

        min_distances = []
        for z in normalized_zeros:
            distances = np.abs(normalized_survivors - z)
            min_distances.append(np.min(distances))

        mean_min_distance = np.mean(min_distances)

        # Expected distance for random uniform: ~1/(2*len(survivors))
        expected_random = p / (2 * len(survivors))

        self.log(f"\nMean minimum distance (normalized): {mean_min_distance:.6f}")
        self.log(f"Expected for random: ~{expected_random/p:.6f}")

        # Correlation exists if observed << expected
        correlation_detected = mean_min_distance < 0.5 * (expected_random / p)

        self.log(f"Correlation detected: {correlation_detected}")

        # Statistical significance test
        # Generate random reference set
        n_random_trials = 1000
        random_distances = []

        for _ in range(n_random_trials):
            random_points = np.random.uniform(0, 1, len(zeta_zeros_gamma))
            trial_min_dists = []
            for rpt in random_points:
                dists = np.abs(normalized_survivors - rpt)
                trial_min_dists.append(np.min(dists))
            random_distances.append(np.mean(trial_min_dists))

        random_distances = np.array(random_distances)
        p_value = np.mean(random_distances <= mean_min_distance)

        self.log(f"p-value (vs random): {p_value:.4f}")
        self.log(f"Statistically significant: {p_value < 0.05}")

        return {
            'test': 'zeta_zero_positions',
            'p': p,
            'mean_min_distance': mean_min_distance,
            'expected_random': expected_random / p,
            'correlation_detected': correlation_detected,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05
        }


def run_extended_analysis(primes: List[int] = None, max_prime: int = 10007):
    """
    Run comprehensive Zeta coupling analysis.

    This pushes computational limits to rigorously investigate
    whether genuine Zeta coupling exists beyond motivation.
    """
    print("="*70)
    print("EXTENDED ZETA COUPLING ANALYSIS")
    print("="*70)
    print("\nObjective: Rigorously test for Riemann Zeta zero coupling")
    print("Approach: Truth over comfort - if no coupling exists, we'll find that")
    print()

    if primes is None:
        # Use primes of increasing size
        primes = [1009, 2003, 5003, 10007]

    tester = ZetaCouplingTests(verbose=True)

    # TEST A: Spectral statistics
    print("\n" + "#"*70)
    print("# SPECTRAL STATISTICS INVESTIGATION")
    print("#"*70)
    result_A = tester.test_spectral_statistics_k4(primes)
    tester.results['spectral_statistics'] = result_A

    # TEST B: Periodic orbits
    print("\n" + "#"*70)
    print("# PERIODIC ORBIT STRUCTURE")
    print("#"*70)
    result_B = tester.test_periodic_orbit_structure(primes[:2], [3, 4, 5, 6])
    tester.results['periodic_orbits'] = result_B

    # TEST C: Prime dependence
    print("\n" + "#"*70)
    print("# PRIME MODULUS SCALING")
    print("#"*70)
    result_C4 = tester.test_prime_modulus_dependence(K=4, primes=primes)
    tester.results['prime_dependence_k4'] = result_C4

    result_C5 = tester.test_prime_modulus_dependence(K=5, primes=primes)
    tester.results['prime_dependence_k5'] = result_C5

    # TEST D: Long-term correlations
    print("\n" + "#"*70)
    print("# FIBER STATE CORRELATIONS")
    print("#"*70)
    result_D = tester.test_fiber_state_correlations(p=10007, K=4, n_cycles=10000)
    tester.results['fiber_correlations'] = result_D

    # TEST E: Critical coupling
    print("\n" + "#"*70)
    print("# CRITICAL COUPLING INVESTIGATION")
    print("#"*70)
    result_E = tester.test_critical_line_analog(primes[:3])
    tester.results['critical_coupling'] = result_E

    # TEST F: Direct Zeta zero correlation
    print("\n" + "#"*70)
    print("# DIRECT ZETA ZERO CORRELATION")
    print("#"*70)
    result_F = tester.test_zeta_zero_positions(p=10007)
    tester.results['zeta_zero_correlation'] = result_F

    # FINAL ASSESSMENT
    print("\n" + "="*70)
    print("CRITICAL ASSESSMENT: ZETA COUPLING")
    print("="*70)

    print("\n1. SPECTRAL STATISTICS:")
    if result_A['suggests_gue_statistics']:
        print("   ⚠ GUE-like statistics detected - suggests quantum chaos connection")
    else:
        print("   ✗ No GUE statistics - appears Poisson (uncorrelated)")

    print("\n2. PERIODIC STRUCTURE:")
    if result_B['k4_period1_dominant']:
        print("   ✓ K=4 dominated by period-1 (fixed points)")
    else:
        print("   ✗ Complex periodic structure at K=4")

    print("\n3. CRITICAL COUPLING:")
    if result_E['k4_is_critical']:
        print("   ✓ K=4 is uniquely critical (analog to Re(s)=1/2)")
    else:
        print("   ✗ K=4 not uniquely critical")

    print("\n4. DIRECT ZETA CORRELATION:")
    if result_F['statistically_significant']:
        print("   ⚠⚠⚠ SIGNIFICANT: Fiber states correlate with Zeta zeros")
        print(f"   p-value: {result_F['p_value']:.6f}")
    else:
        print("   ✗ No significant correlation with Zeta zero positions")
        print(f"   p-value: {result_F['p_value']:.4f}")

    print("\n5. LONG-TERM CORRELATIONS:")
    if result_D['is_constant']:
        print("   ✓ K=4 produces constant trajectories (perfect memory)")
    else:
        print("   Note: K=4 trajectories should be constant")

    # Overall verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    evidence_for_coupling = [
        result_A['suggests_gue_statistics'],
        result_E['k4_is_critical'],
        result_F['statistically_significant']
    ]

    strong_evidence_count = sum(evidence_for_coupling)

    if strong_evidence_count >= 2:
        print("\n⚠⚠⚠ SIGNIFICANT EVIDENCE FOR ZETA COUPLING")
        print("Multiple independent tests suggest non-trivial connection")
        print("Further investigation warranted with:")
        print("  - Larger primes (p > 10^6)")
        print("  - More Zeta zeros")
        print("  - Explicit formula analysis")
    elif strong_evidence_count == 1:
        print("\n⚠ WEAK EVIDENCE FOR ZETA COUPLING")
        print("Some suggestive patterns, but not definitive")
        print("Connection remains heuristic pending further tests")
    else:
        print("\n✗ NO EVIDENCE FOR DIRECT ZETA COUPLING")
        print("System exhibits arithmetic structure but no proven")
        print("connection to Riemann Zeta zero dynamics")
        print("\nConclusion: Zeta reference is motivational analogy only")

    print()

    return tester.results


if __name__ == "__main__":
    # Run with progressively larger primes
    if len(sys.argv) > 1:
        max_prime = int(sys.argv[1])
    else:
        max_prime = 10007

    # Prime sequence
    test_primes = [1009, 2003, 5003, 10007, 20011, 50021, 100003]
    test_primes = [p for p in test_primes if p <= max_prime]

    print(f"Testing with primes up to {max_prime}")
    print(f"Prime sequence: {test_primes}")
    print()

    results = run_extended_analysis(primes=test_primes)
