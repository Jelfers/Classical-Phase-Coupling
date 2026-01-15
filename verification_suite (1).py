#!/usr/bin/env python3
"""
Collatz-Zeta Verification Suite
================================

Rigorous verification of the skew-product Collatz dynamical system
coupled with modular arithmetic. This suite tests the mathematical
claims regarding K=4 invariance and K≠4 extinction behavior.

Mathematical System:
-------------------
T : Z × Z_p → Z × Z_p

Even step: (w, n) ↦ (w/2, n/2 mod p)
Odd step:  (w, n) ↦ (3w + 1 + c, Kn mod p)
           where c = ⌊Kn/p⌋

Key Claims to Verify:
--------------------
1. K=4: Exact invariance over Collatz cycle {1→4→2→1}
2. K≠4: Multiplicative sieve leads to rapid extinction
3. Carry gate condition: c=0 ⟺ Kn < p
4. Safe window: W_K = [0, (p-1)/K]
5. Return map: R_K(n) ≡ (K/4)n mod p
"""

import numpy as np
from typing import Tuple, List, Set, Dict
from dataclasses import dataclass
from collections import defaultdict
import sys


@dataclass
class TrajectoryResult:
    """Result of trajectory computation."""
    survived: bool
    steps: int
    ejection_step: int | None
    ejection_cause: str | None
    trajectory: List[Tuple[int, int]]


class CollatzSkewProduct:
    """
    Implementation of the skew-product Collatz dynamical system.

    Parameters:
    -----------
    K : int
        Fiber multiplication constant
    p : int
        Prime modulus for fiber arithmetic
    """

    def __init__(self, K: int, p: int):
        if p <= K:
            raise ValueError(f"Prime p={p} must be greater than K={K}")
        self.K = K
        self.p = p
        self.safe_window = (p - 1) // K

    def carry_term(self, n: int) -> int:
        """
        Compute carry term c = ⌊Kn/p⌋

        This is the ONLY coupling between base and fiber.
        """
        return (self.K * n) // self.p

    def is_in_safe_window(self, n: int) -> bool:
        """
        Check if n is in the safe window W_K = [0, (p-1)/K]
        """
        return 0 <= n <= self.safe_window

    def step(self, w: int, n: int) -> Tuple[int, int, int]:
        """
        Execute one step of the skew-product system.

        Returns:
        --------
        (w_next, n_next, carry)
        """
        if w % 2 == 0:
            # Even step: no carry
            w_next = w // 2
            n_next = (n * pow(2, -1, self.p)) % self.p
            carry = 0
        else:
            # Odd step: carry injection
            carry = self.carry_term(n)
            w_next = 3 * w + 1 + carry
            n_next = (self.K * n) % self.p

        return w_next, n_next, carry

    def evolve_trajectory(self, w0: int, n0: int, max_steps: int = 10000,
                         check_cycle: bool = True) -> TrajectoryResult:
        """
        Evolve trajectory and detect ejection from {1,4,2} cycle.

        Ejection occurs when:
        1. carry ≠ 0 at w=1 (gate violation)
        2. trajectory leaves {1,4,2,1} set
        3. trajectory diverges beyond tracking
        """
        w, n = w0, n0
        trajectory = [(w, n)]
        collatz_cycle = {1, 4, 2}

        for step in range(max_steps):
            w_next, n_next, carry = self.step(w, n)
            trajectory.append((w_next, n_next))

            # Check for gate violation at w=1
            if check_cycle and w == 1 and carry != 0:
                return TrajectoryResult(
                    survived=False,
                    steps=step + 1,
                    ejection_step=step,
                    ejection_cause=f"carry_gate_violation: c={carry} at w=1",
                    trajectory=trajectory
                )

            # Check if ejected from Collatz cycle
            if check_cycle and w in collatz_cycle and w_next not in collatz_cycle:
                return TrajectoryResult(
                    survived=False,
                    steps=step + 1,
                    ejection_step=step,
                    ejection_cause=f"cycle_ejection: {w}→{w_next}",
                    trajectory=trajectory
                )

            # Check for divergence
            if w_next > 1e10:
                return TrajectoryResult(
                    survived=False,
                    steps=step + 1,
                    ejection_step=step,
                    ejection_cause="divergence",
                    trajectory=trajectory
                )

            w, n = w_next, n_next

        # Survived max_steps
        return TrajectoryResult(
            survived=True,
            steps=max_steps,
            ejection_step=None,
            ejection_cause=None,
            trajectory=trajectory
        )

    def compute_return_map(self, n: int) -> int:
        """
        Compute fiber state after one full Collatz cycle {1→4→2→1}.

        Mathematically: R_K(n) ≡ (K/4)n mod p
        """
        # 1 → 4: n ↦ Kn
        n1 = (self.K * n) % self.p

        # 4 → 2: n ↦ n/2
        n2 = (n1 * pow(2, -1, self.p)) % self.p

        # 2 → 1: n ↦ n/2
        n3 = (n2 * pow(2, -1, self.p)) % self.p

        return n3

    def verify_return_map_formula(self, n: int) -> Tuple[int, int, bool]:
        """
        Verify that computed return map matches theoretical formula.

        Returns: (computed, theoretical, match)
        """
        computed = self.compute_return_map(n)

        # Theoretical: R_K(n) = (K/4)n mod p
        K_over_4 = (self.K * pow(4, -1, self.p)) % self.p
        theoretical = (K_over_4 * n) % self.p

        return computed, theoretical, (computed == theoretical)


class VerificationTests:
    """
    Comprehensive test suite for mathematical claims.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def test_carry_gate_condition(self, K: int, p: int, n_samples: int = 1000) -> Dict:
        """
        TEST 1: Verify carry gate condition c=0 ⟺ Kn < p
        """
        self.log(f"\n{'='*70}")
        self.log(f"TEST 1: Carry Gate Condition (K={K}, p={p})")
        self.log(f"{'='*70}")

        system = CollatzSkewProduct(K, p)
        failures = []

        # Test across full range
        test_points = np.linspace(0, p-1, n_samples, dtype=int)

        for n in test_points:
            c = system.carry_term(n)
            in_window = system.is_in_safe_window(n)

            # Verify: c=0 ⟺ n in safe window
            if (c == 0) != in_window:
                failures.append({
                    'n': n,
                    'carry': c,
                    'in_window': in_window,
                    'Kn': K * n,
                    'p': p
                })

        passed = len(failures) == 0
        self.log(f"Samples tested: {n_samples}")
        self.log(f"Safe window: [0, {system.safe_window}]")
        self.log(f"Window width: {system.safe_window + 1} values")
        self.log(f"Window fraction: {(system.safe_window + 1) / p:.6f}")
        self.log(f"Failures: {len(failures)}")
        self.log(f"Status: {'PASS' if passed else 'FAIL'}")

        return {
            'test': 'carry_gate_condition',
            'K': K,
            'p': p,
            'passed': passed,
            'failures': failures,
            'safe_window_size': system.safe_window + 1,
            'window_fraction': (system.safe_window + 1) / p
        }

    def test_return_map_formula(self, K: int, p: int, n_samples: int = 1000) -> Dict:
        """
        TEST 2: Verify return map R_K(n) ≡ (K/4)n mod p
        """
        self.log(f"\n{'='*70}")
        self.log(f"TEST 2: Return Map Formula (K={K}, p={p})")
        self.log(f"{'='*70}")

        system = CollatzSkewProduct(K, p)
        failures = []

        test_points = np.linspace(0, p-1, n_samples, dtype=int)

        for n in test_points:
            computed, theoretical, match = system.verify_return_map_formula(n)
            if not match:
                failures.append({
                    'n': n,
                    'computed': computed,
                    'theoretical': theoretical
                })

        passed = len(failures) == 0
        self.log(f"Samples tested: {n_samples}")
        self.log(f"Formula: R_{K}(n) ≡ ({K}/4)n mod {p}")
        self.log(f"Failures: {len(failures)}")
        self.log(f"Status: {'PASS' if passed else 'FAIL'}")

        return {
            'test': 'return_map_formula',
            'K': K,
            'p': p,
            'passed': passed,
            'failures': failures
        }

    def test_k4_invariance(self, p: int, n_samples: int = 100, max_steps: int = 1000) -> Dict:
        """
        TEST 3: Verify K=4 exact invariance

        For K=4: R_4(n) ≡ n mod p (identity map)
        All points in safe window should survive indefinitely.
        """
        self.log(f"\n{'='*70}")
        self.log(f"TEST 3: K=4 Exact Invariance (p={p})")
        self.log(f"{'='*70}")

        K = 4
        system = CollatzSkewProduct(K, p)

        # First verify return map is identity
        self.log("\nVerifying R_4(n) = n for n in safe window...")
        identity_failures = []

        test_points = np.linspace(0, system.safe_window,
                                 min(n_samples, system.safe_window + 1),
                                 dtype=int)

        for n in test_points:
            n_return = system.compute_return_map(n)
            if n_return != n:
                identity_failures.append({
                    'n': n,
                    'returned': n_return
                })

        identity_pass = len(identity_failures) == 0
        self.log(f"Identity check: {'PASS' if identity_pass else 'FAIL'}")
        self.log(f"Identity failures: {len(identity_failures)}")

        # Now test trajectory survival
        self.log(f"\nTesting trajectory survival (w0=1, {len(test_points)} samples, {max_steps} steps)...")
        survival_results = []
        ejections = []

        for n in test_points:
            result = system.evolve_trajectory(1, n, max_steps=max_steps)
            survival_results.append(result)

            if not result.survived:
                ejections.append({
                    'n': n,
                    'ejection_step': result.ejection_step,
                    'cause': result.ejection_cause
                })

        survival_rate = sum(1 for r in survival_results if r.survived) / len(survival_results)
        survival_pass = survival_rate == 1.0

        self.log(f"Trajectories tested: {len(survival_results)}")
        self.log(f"Survivors: {sum(1 for r in survival_results if r.survived)}")
        self.log(f"Ejections: {len(ejections)}")
        self.log(f"Survival rate: {survival_rate:.4f}")
        self.log(f"Status: {'PASS' if (identity_pass and survival_pass) else 'FAIL'}")

        # Show first few ejections if any
        if ejections:
            self.log(f"\nFirst ejections:")
            for ej in ejections[:5]:
                self.log(f"  n={ej['n']}: step {ej['ejection_step']}, {ej['cause']}")

        return {
            'test': 'k4_invariance',
            'K': 4,
            'p': p,
            'passed': identity_pass and survival_pass,
            'identity_check': {
                'passed': identity_pass,
                'failures': identity_failures
            },
            'trajectory_survival': {
                'passed': survival_pass,
                'survival_rate': survival_rate,
                'ejections': ejections
            }
        }

    def test_k_not_4_extinction(self, K_values: List[int], p: int,
                                n_samples: int = 50, max_steps: int = 1000) -> Dict:
        """
        TEST 4: Verify K≠4 leads to rapid extinction

        For K≠4, multiplicative sieve should cause ejection.
        """
        self.log(f"\n{'='*70}")
        self.log(f"TEST 4: K≠4 Extinction Behavior (p={p})")
        self.log(f"{'='*70}")

        results_by_k = {}

        for K in K_values:
            if K == 4:
                continue  # Skip K=4

            self.log(f"\nTesting K={K}...")
            system = CollatzSkewProduct(K, p)

            # Sample from safe window
            window_size = min(system.safe_window + 1, n_samples)
            test_points = np.linspace(0, system.safe_window, window_size, dtype=int)

            survival_results = []
            ejection_times = []

            for n in test_points:
                result = system.evolve_trajectory(1, n, max_steps=max_steps)
                survival_results.append(result)

                if not result.survived and result.ejection_step is not None:
                    ejection_times.append(result.ejection_step)

            survivors = sum(1 for r in survival_results if r.survived)
            survival_rate = survivors / len(survival_results)
            mean_ejection = np.mean(ejection_times) if ejection_times else float('inf')
            median_ejection = np.median(ejection_times) if ejection_times else float('inf')

            self.log(f"  Safe window size: {system.safe_window + 1}")
            self.log(f"  Samples tested: {len(test_points)}")
            self.log(f"  Survivors: {survivors}")
            self.log(f"  Ejections: {len(ejection_times)}")
            self.log(f"  Survival rate: {survival_rate:.4f}")
            self.log(f"  Mean ejection time: {mean_ejection:.1f}")
            self.log(f"  Median ejection time: {median_ejection:.1f}")

            results_by_k[K] = {
                'K': K,
                'safe_window_size': system.safe_window + 1,
                'samples_tested': len(test_points),
                'survivors': survivors,
                'survival_rate': survival_rate,
                'mean_ejection_time': mean_ejection,
                'median_ejection_time': median_ejection,
                'ejection_times': ejection_times
            }

        # Overall assessment: expect very low survival for K≠4
        all_survival_rates = [r['survival_rate'] for r in results_by_k.values()]
        max_survival_rate = max(all_survival_rates) if all_survival_rates else 0
        passed = max_survival_rate < 0.1  # Less than 10% survival for all K≠4

        self.log(f"\n{'='*50}")
        self.log(f"Maximum survival rate across all K≠4: {max_survival_rate:.4f}")
        self.log(f"Status: {'PASS' if passed else 'FAIL'}")

        return {
            'test': 'k_not_4_extinction',
            'p': p,
            'passed': passed,
            'results_by_k': results_by_k,
            'max_survival_rate': max_survival_rate
        }

    def test_return_map_structure(self, K_values: List[int], p: int, n_samples: int = 100) -> Dict:
        """
        TEST 5: Analyze return map multiplicative structure

        For K≠4: R_K^t(n) should eventually exit safe window
        """
        self.log(f"\n{'='*70}")
        self.log(f"TEST 5: Return Map Multiplicative Structure (p={p})")
        self.log(f"{'='*70}")

        results_by_k = {}

        for K in K_values:
            self.log(f"\nAnalyzing K={K}...")
            system = CollatzSkewProduct(K, p)

            # Compute return map multiplier
            if K == 4:
                multiplier_str = "1 (identity)"
            else:
                K_over_4 = (K * pow(4, -1, p)) % p
                multiplier_str = f"{K_over_4}"

            self.log(f"  Return multiplier R_{K} ≡ {multiplier_str} × n (mod {p})")

            # Sample orbits under iterated return map
            test_points = np.linspace(0, min(system.safe_window, n_samples-1),
                                     min(n_samples, system.safe_window + 1),
                                     dtype=int)

            orbit_lengths = []
            max_iterations = 1000

            for n0 in test_points:
                if n0 == 0:
                    continue

                n = n0
                for t in range(max_iterations):
                    n = system.compute_return_map(n)

                    # Check if exited safe window
                    if not system.is_in_safe_window(n):
                        orbit_lengths.append(t + 1)
                        break
                else:
                    # Never exited in max_iterations
                    orbit_lengths.append(max_iterations)

            mean_orbit = np.mean(orbit_lengths) if orbit_lengths else 0
            median_orbit = np.median(orbit_lengths) if orbit_lengths else 0
            max_orbit = max(orbit_lengths) if orbit_lengths else 0

            # Count how many stayed in window full time
            long_survivors = sum(1 for l in orbit_lengths if l >= max_iterations)

            self.log(f"  Orbits analyzed: {len(orbit_lengths)}")
            self.log(f"  Mean orbit length: {mean_orbit:.1f}")
            self.log(f"  Median orbit length: {median_orbit:.1f}")
            self.log(f"  Max orbit length: {max_orbit}")
            self.log(f"  Long survivors (>{max_iterations//2} iterations): {long_survivors}")

            results_by_k[K] = {
                'K': K,
                'multiplier': multiplier_str,
                'mean_orbit_length': mean_orbit,
                'median_orbit_length': median_orbit,
                'max_orbit_length': max_orbit,
                'long_survivors': long_survivors,
                'orbit_lengths': orbit_lengths
            }

        # K=4 should have all long survivors
        k4_check = results_by_k.get(4, {}).get('long_survivors', 0)
        k4_total = len(results_by_k.get(4, {}).get('orbit_lengths', [1]))
        k4_passed = (k4_check / k4_total > 0.95) if k4_total > 0 else False

        self.log(f"\n{'='*50}")
        self.log(f"K=4 long survivors: {k4_check}/{k4_total} = {k4_check/k4_total:.2%}")
        self.log(f"Status: {'PASS' if k4_passed else 'FAIL'}")

        return {
            'test': 'return_map_structure',
            'p': p,
            'passed': k4_passed,
            'results_by_k': results_by_k
        }


def run_comprehensive_verification(prime: int = 1009, n_samples: int = 100):
    """
    Run full verification suite with specified parameters.

    Parameters:
    -----------
    prime : int
        Prime modulus (default: 1009, a moderate-sized prime)
    n_samples : int
        Number of samples per test
    """
    print("="*70)
    print("COLLATZ-ZETA VERIFICATION SUITE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Prime modulus p = {prime}")
    print(f"  Samples per test = {n_samples}")
    print(f"\nObjective: Falsify or verify mathematical claims")
    print(f"  - K=4: Exact invariance (expect survival)")
    print(f"  - K≠4: Multiplicative sieve (expect extinction)")
    print()

    tester = VerificationTests(verbose=True)

    # Test suite
    K_test_values = [2, 3, 4, 5, 6, 8, 12]

    # TEST 1: Carry gate condition (multiple K values)
    for K in [3, 4, 5, 8]:
        result = tester.test_carry_gate_condition(K, prime, n_samples=n_samples)
        tester.results[f'carry_gate_K{K}'] = result

    # TEST 2: Return map formula (multiple K values)
    for K in K_test_values:
        result = tester.test_return_map_formula(K, prime, n_samples=n_samples)
        tester.results[f'return_map_K{K}'] = result

    # TEST 3: K=4 invariance (critical test)
    result = tester.test_k4_invariance(prime, n_samples=n_samples, max_steps=1000)
    tester.results['k4_invariance'] = result

    # TEST 4: K≠4 extinction
    result = tester.test_k_not_4_extinction(
        K_values=[k for k in K_test_values if k != 4],
        p=prime,
        n_samples=n_samples // 2,
        max_steps=500
    )
    tester.results['k_not_4_extinction'] = result

    # TEST 5: Return map structure
    result = tester.test_return_map_structure(K_test_values, prime, n_samples=n_samples)
    tester.results['return_map_structure'] = result

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed_tests = sum(1 for r in tester.results.values() if r.get('passed', False))
    total_tests = len(tester.results)

    print(f"\nTests passed: {passed_tests}/{total_tests}")
    print(f"\nDetailed results:")

    for test_name, result in tester.results.items():
        status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        print(f"  {status} - {test_name}")

    # Critical assessment
    print("\n" + "="*70)
    print("CRITICAL ASSESSMENT")
    print("="*70)

    k4_passed = tester.results.get('k4_invariance', {}).get('passed', False)
    extinction_passed = tester.results.get('k_not_4_extinction', {}).get('passed', False)

    print(f"\nCore claims:")
    print(f"  1. K=4 exact invariance: {'VERIFIED' if k4_passed else 'FALSIFIED'}")
    print(f"  2. K≠4 rapid extinction: {'VERIFIED' if extinction_passed else 'FALSIFIED'}")

    if k4_passed and extinction_passed:
        print(f"\n→ Mathematical structure confirmed: K=4 is arithmetically unique.")
    elif not k4_passed:
        print(f"\n→ K=4 invariance claim FALSIFIED - mechanism requires revision.")
    elif not extinction_passed:
        print(f"\n→ K≠4 extinction weaker than claimed - sieve less selective.")

    print()

    return tester.results


if __name__ == "__main__":
    # Default run
    if len(sys.argv) > 1:
        prime = int(sys.argv[1])
    else:
        prime = 1009  # Moderate-sized prime for testing

    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    else:
        n_samples = 100

    results = run_comprehensive_verification(prime=prime, n_samples=n_samples)
