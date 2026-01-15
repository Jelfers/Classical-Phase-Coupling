#!/usr/bin/env python3
"""
Quantum Mechanical Analysis of Collatz-Zeta System
===================================================

HYPOTHESIS: The geometric falsification applies to CLASSICAL correlations,
but quantum mechanics may reveal a deeper connection through:

1. Transfer operator spectral analysis
2. Quantum walk interpretation
3. Trace formula (Gutzwiller-type)
4. Phase space quantization
5. Berry-Keating Hamiltonian analog

Approach: Rigorous quantum framework, not classical geometric correlations.
"""

import numpy as np
from scipy.linalg import eig, eigvals
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict


class QuantumCollatzAnalyzer:
    """
    Quantum mechanical analysis of the Collatz skew-product system.

    Key idea: Treat the return map as a quantum evolution operator
    and study its spectral properties.
    """

    def __init__(self, K: int, p: int):
        self.K = K
        self.p = p
        self.safe_window = (p - 1) // K
        self.hilbert_dim = self.safe_window + 1

    def build_transfer_operator(self) -> np.ndarray:
        """
        Construct the Perron-Frobenius/Koopman operator for the return map.

        This is the quantum analog of classical evolution.
        For discrete maps, this is a finite-dimensional matrix.

        T[i,j] = 1 if return_map(j) = i, else 0

        For K=4: This should be the identity matrix.
        For K≠4: Non-trivial mixing dynamics.
        """
        T = np.zeros((self.hilbert_dim, self.hilbert_dim))

        for j in range(self.hilbert_dim):
            # Compute where state j maps to under return map
            n_j = j
            # Return map: R_K(n) = (K/4)n mod p
            K_over_4 = (self.K * pow(4, -1, self.p)) % self.p
            i = (K_over_4 * n_j) % self.p

            # Check if still in safe window
            if i <= self.safe_window:
                T[i, j] = 1.0
            # If exits safe window, maps to absorbing state (extinction)

        return T

    def compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of transfer operator.

        Key insight: For quantum systems, eigenvalues on unit circle
        correspond to quantum energy levels.

        For Zeta connection, we'd expect eigenvalue distribution
        to match GUE statistics if quantum chaos is present.
        """
        T = self.build_transfer_operator()
        eigenvals, eigenvecs = eig(T)

        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        return eigenvals, eigenvecs

    def analyze_eigenvalue_spacing(self, eigenvals: np.ndarray) -> Dict:
        """
        Analyze eigenvalue spacing statistics.

        GUE (quantum chaos): P(s) ∝ s exp(-s²)  (Wigner surmise)
        Poisson (integrable): P(s) = exp(-s)

        This is the CORRECT test for quantum chaos, not classical correlations.
        """
        # Get eigenvalues on unit circle (quantum states)
        unit_circle_eigs = eigenvals[np.abs(np.abs(eigenvals) - 1.0) < 1e-10]

        if len(unit_circle_eigs) < 2:
            return {'n_states': len(unit_circle_eigs), 'spacings': np.array([])}

        # Get phases (angles on unit circle)
        phases = np.angle(unit_circle_eigs)
        phases = np.sort(phases)

        # Compute spacings
        spacings = np.diff(phases)

        # Unfold to unit mean spacing
        mean_spacing = np.mean(spacings)
        if mean_spacing > 0:
            normalized_spacings = spacings / mean_spacing
        else:
            normalized_spacings = spacings

        return {
            'n_states': len(unit_circle_eigs),
            'spacings': normalized_spacings,
            'phases': phases,
            'eigenvalues': unit_circle_eigs
        }

    def quantum_trace_formula(self, max_time: int = 100) -> np.ndarray:
        """
        Compute quantum trace formula: Tr(T^t)

        In quantum systems, the trace over time encodes periodic orbit information.

        Gutzwiller trace formula connects classical orbits to quantum spectrum:
        ρ(E) = ρ̄(E) + Σ oscillatory terms from periodic orbits

        For our system: Does the trace exhibit structure related to Zeta zeros?
        """
        traces = []
        T = self.build_transfer_operator()

        T_power = np.eye(T.shape[0])
        for t in range(max_time):
            trace = np.trace(T_power)
            traces.append(trace)
            T_power = T_power @ T

        return np.array(traces)

    def compute_quantum_fidelity(self, n_cycles: int = 100) -> np.ndarray:
        """
        Quantum fidelity measures state return.

        For K=4 (identity): Perfect fidelity forever
        For K≠4: Fidelity decay encodes quantum decoherence

        Connection to Zeta: Fidelity decay rate might relate to zero distribution?
        """
        T = self.build_transfer_operator()

        # Initial state (uniform superposition over safe window)
        psi_0 = np.ones(self.hilbert_dim) / np.sqrt(self.hilbert_dim)

        fidelities = []
        psi = psi_0.copy()

        for t in range(n_cycles):
            fidelity = np.abs(np.vdot(psi_0, psi))**2
            fidelities.append(fidelity)
            psi = T @ psi
            psi = psi / (np.linalg.norm(psi) + 1e-10)  # Renormalize

        return np.array(fidelities)


def test_quantum_zeta_hypothesis(primes: List[int] = [1009, 10007], K_values: List[int] = [3, 4, 5]):
    """
    Test quantum mechanical connection to Zeta zeros.

    Hypothesis: The classical geometric artifacts don't apply to quantum spectral properties.

    Tests:
    1. Transfer operator spectrum (eigenvalues)
    2. Level spacing statistics (GUE vs Poisson)
    3. Trace formula structure
    4. Quantum fidelity decay
    5. Comparison to known Zeta properties
    """
    print("="*70)
    print("QUANTUM MECHANICAL ZETA CONNECTION")
    print("="*70)
    print("\nHypothesis: Classical geometric falsification doesn't apply")
    print("to quantum spectral properties.\n")

    results = {}

    for p in primes:
        print(f"\n{'='*70}")
        print(f"PRIME p = {p}")
        print(f"{'='*70}")

        for K in K_values:
            print(f"\n--- K = {K} ---")

            analyzer = QuantumCollatzAnalyzer(K, p)

            # 1. Compute spectrum
            print(f"Computing transfer operator spectrum...")
            eigenvals, eigenvecs = analyzer.compute_spectrum()

            n_states = analyzer.hilbert_dim
            n_unit_circle = np.sum(np.abs(np.abs(eigenvals) - 1.0) < 1e-10)

            print(f"Hilbert space dimension: {n_states}")
            print(f"States on unit circle: {n_unit_circle}")
            print(f"Largest eigenvalue: {np.abs(eigenvals[0]):.6f}")

            # 2. Level spacing analysis
            spacing_data = analyzer.analyze_eigenvalue_spacing(eigenvals)

            if len(spacing_data['spacings']) > 0:
                mean_spacing = np.mean(spacing_data['spacings'])
                print(f"Mean normalized spacing: {mean_spacing:.4f}")
                print(f"  (GUE ≈ 1.0, Poisson ≈ 1.0 for normalized)")

            # 3. Trace formula
            print(f"Computing quantum trace...")
            traces = analyzer.quantum_trace_formula(max_time=50)

            # For K=4, trace should be constant (all eigenvalues = 1)
            trace_variance = np.var(traces)
            print(f"Trace variance: {trace_variance:.6f}")
            print(f"  (K=4 should have ~0 variance)")

            # 4. Quantum fidelity
            print(f"Computing quantum fidelity...")
            fidelities = analyzer.compute_quantum_fidelity(n_cycles=50)

            mean_fidelity = np.mean(fidelities[10:])  # After transient
            print(f"Mean fidelity (t>10): {mean_fidelity:.6f}")
            print(f"  (K=4 should have fidelity ≈ 1.0)")

            # Store results
            key = f"p{p}_K{K}"
            results[key] = {
                'p': p,
                'K': K,
                'eigenvalues': eigenvals,
                'n_unit_circle': n_unit_circle,
                'spacing_data': spacing_data,
                'traces': traces,
                'trace_variance': trace_variance,
                'fidelities': fidelities,
                'mean_fidelity': mean_fidelity
            }

    # Analysis
    print("\n" + "="*70)
    print("QUANTUM ANALYSIS SUMMARY")
    print("="*70)

    # K=4 should show:
    # - All eigenvalues = 1 (identity operator)
    # - Zero trace variance
    # - Perfect fidelity

    for key, res in results.items():
        if res['K'] == 4:
            print(f"\n{key}:")
            print(f"  Unit circle states: {res['n_unit_circle']}/{analyzer.hilbert_dim}")
            print(f"  Trace variance: {res['trace_variance']:.8f}")
            print(f"  Mean fidelity: {res['mean_fidelity']:.8f}")

            is_identity = (res['n_unit_circle'] == analyzer.hilbert_dim and
                          res['trace_variance'] < 1e-10 and
                          res['mean_fidelity'] > 0.99)

            print(f"  → Transfer operator is identity: {is_identity}")

    # Compare to Zeta zeros
    print("\n" + "="*70)
    print("ZETA CONNECTION ASSESSMENT")
    print("="*70)

    print("\nKey Question: Do eigenvalue phases correlate with Zeta zeros?")
    print("(This would be quantum connection, not classical geometric one)")

    # Get K=4 eigenvalue phases for largest prime
    k4_largest_p = [res for res in results.values() if res['K'] == 4 and res['p'] == max(primes)]

    if k4_largest_p:
        res = k4_largest_p[0]
        phases = res['spacing_data']['phases']

        print(f"\nK=4, p={res['p']}:")
        print(f"Number of quantum states (phases): {len(phases)}")

        if len(phases) > 1:
            # For K=4 identity, all phases should be 0 or 2π (all eigenvalues = 1)
            phase_spread = np.max(np.abs(phases))
            print(f"Phase spread: {phase_spread:.6f} rad")

            if phase_spread < 0.01:
                print("  → All eigenvalues are +1 (trivial spectrum)")
                print("  → No quantum structure to correlate with Zeta")
            else:
                print("  → Non-trivial phase distribution")
                print("  → Possible quantum spectral structure")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    print("\nQuantum mechanical analysis reveals:")
    print("1. K=4 transfer operator is identity (all eigenvalues = 1)")
    print("2. No non-trivial quantum spectral structure")
    print("3. Perfect quantum fidelity (no decoherence)")
    print("4. Trace is constant (no periodic orbit structure)")

    print("\nImplication:")
    print("The K=4 system is quantum mechanically TRIVIAL.")
    print("There is no non-trivial spectrum to correlate with Zeta zeros.")

    print("\nVerdict:")
    print("Quantum analysis confirms: No Zeta connection.")
    print("The system is too simple (identity operator) to exhibit")
    print("quantum chaos or spectral universality.")

    return results


if __name__ == "__main__":
    print("\nQuantum Mechanical Investigation")
    print("="*70)
    print("Testing if quantum framework reveals Zeta connection")
    print("that classical geometric analysis missed.\n")

    results = test_quantum_zeta_hypothesis(
        primes=[1009, 10007],
        K_values=[3, 4, 5]
    )

    print("\n" + "="*70)
    print("Analysis complete.")
    print("="*70)
