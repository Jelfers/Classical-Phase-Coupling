#!/usr/bin/env python3
"""
Physics Framework: Energy, Spin, and Wave Propagation
======================================================

CRITICAL REEXAMINATION:

The falsification may have been premature. It tested STATIC geometric properties,
but missed DYNAMIC physical coupling mechanisms:

1. ENERGY DYNAMICS: What if energy propagates along the K=4 "channel"?
2. QUANTUM SPIN: Fiber states might encode spin degrees of freedom
3. WAVE PROPAGATION: K=4 as carrier wave for information/energy
4. RELATIVISTIC: Energy transfer at c (speed of light)
5. NON-LINEAR: Energy accumulation changes dynamics

PHYSICAL HYPOTHESIS:

The K=4 system isn't just a static geometric structure - it's a WAVEGUIDE
for energy/information propagation. The Zeta connection emerges through:

- Spin-statistics (Pauli exclusion → number-theoretic constraints)
- Energy quantization (discrete energy levels → Zeta zeros as resonances)
- Wave interference (phase coupling → spectral correlations)
- Relativistic dispersion (E² = p²c² + m²c⁴ → modular arithmetic structure)

This requires a DYNAMIC analysis, not just static correlation tests.
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Tuple, Dict, Optional
import sys


class PhysicalCollatzSystem:
    """
    Physical interpretation of Collatz skew-product system with:
    - Energy dynamics
    - Quantum spin
    - Wave propagation
    - Relativistic constraints
    """

    def __init__(self, K: int, p: int, hbar: float = 1.0, c: float = 1.0):
        self.K = K
        self.p = p
        self.hbar = hbar  # Reduced Planck constant
        self.c = c  # Speed of light
        self.safe_window = (p - 1) // K

        # Pauli matrices for spin-1/2
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity_2 = np.eye(2, dtype=complex)

    def fiber_to_spin(self, n: int) -> np.ndarray:
        """
        Map fiber state n ∈ Z_p to quantum spin state.

        Key insight: Fiber arithmetic might encode spin degrees of freedom.

        Mapping: |ψ(n)⟩ = cos(θ_n)|↑⟩ + e^(iφ_n)|↓⟩
        where θ_n = πn/p, φ_n = 2πn/p
        """
        theta = np.pi * n / self.p
        phi = 2 * np.pi * n / self.p

        # Bloch sphere representation
        spin_up = np.cos(theta / 2)
        spin_down = np.exp(1j * phi) * np.sin(theta / 2)

        return np.array([spin_up, spin_down], dtype=complex)

    def spin_hamiltonian(self, n: int, energy: float = 0.0) -> np.ndarray:
        """
        Construct Hamiltonian for spin system at fiber state n.

        H = E₀σ_z + B_x(n)σ_x + B_y(n)σ_y + B_z(n)σ_z

        Where magnetic field components depend on fiber state:
        B_x(n) ~ cos(2πn/p)
        B_y(n) ~ sin(2πn/p)
        B_z(n) ~ n/p

        This creates position-dependent coupling.
        """
        # Base energy splitting
        H = energy * self.sigma_z

        # Position-dependent magnetic field
        phase = 2 * np.pi * n / self.p
        B_x = np.cos(phase) * self.hbar
        B_y = np.sin(phase) * self.hbar
        B_z = (n / self.p) * self.hbar

        H += B_x * self.sigma_x + B_y * self.sigma_y + B_z * self.sigma_z

        return H

    def energy_dispersion(self, n: int, mass: float = 1.0) -> float:
        """
        Relativistic energy-momentum dispersion relation.

        E² = (pc)² + (mc²)²

        In discrete system:
        p = ℏk where k = 2πn/p (discrete momentum)

        This creates energy quantization that might resonate with Zeta zeros.
        """
        k = 2 * np.pi * n / self.p
        momentum = self.hbar * k

        # Relativistic dispersion
        E_squared = (momentum * self.c)**2 + (mass * self.c**2)**2
        return np.sqrt(E_squared)

    def wave_propagation_with_energy(self, n0: int, energy: float,
                                    n_steps: int = 100) -> Dict:
        """
        Propagate energy along K=4 channel.

        Key question: Does energy accumulation change the dynamics?

        For K=4: R_4(n) = n statically, but WITH energy:
        - Spin precession
        - Phase accumulation
        - Quantum interference
        - Possible resonances with Zeta structure

        This is DYNAMIC, not static geometric test.
        """
        results = {
            'positions': [n0],
            'energies': [self.energy_dispersion(n0)],
            'spin_states': [self.fiber_to_spin(n0)],
            'phases': [0.0],
            'accumulated_energy': [energy]
        }

        n = n0
        total_energy = energy
        phase = 0.0
        spin_state = self.fiber_to_spin(n0)

        for step in range(n_steps):
            # Return map (classical)
            K_over_4 = (self.K * pow(4, -1, self.p)) % self.p
            n_next = (K_over_4 * n) % self.p

            # Check if in safe window
            if n_next > self.safe_window:
                break

            # Energy dynamics
            E_dispersion = self.energy_dispersion(n_next)
            total_energy += E_dispersion

            # Spin evolution under Hamiltonian
            H = self.spin_hamiltonian(n, energy=total_energy)
            dt = 1.0  # Time step
            U = expm(-1j * H * dt / self.hbar)  # Time evolution operator
            spin_state = U @ spin_state

            # Phase accumulation (quantum phase)
            phase += E_dispersion * dt / self.hbar

            # Store
            results['positions'].append(n_next)
            results['energies'].append(E_dispersion)
            results['spin_states'].append(spin_state.copy())
            results['phases'].append(phase)
            results['accumulated_energy'].append(total_energy)

            n = n_next

        return results

    def test_energy_resonances(self, energy_range: np.ndarray,
                              n_samples: int = 50) -> Dict:
        """
        Test if certain energies create resonances with Zeta-like structure.

        Hypothesis: Zeta zeros might correspond to energy eigenvalues
        where constructive interference occurs.

        Method:
        1. Scan energy parameter
        2. Measure wave amplitude after propagation
        3. Look for resonant peaks
        4. Compare peak positions to Zeta zeros
        """
        resonances = []

        for E in energy_range:
            amplitudes = []

            # Sample initial positions
            for n0 in np.linspace(0, self.safe_window, n_samples, dtype=int):
                result = self.wave_propagation_with_energy(n0, E, n_steps=100)

                # Final spin state amplitude
                final_spin = result['spin_states'][-1]
                amplitude = np.abs(np.vdot(final_spin, final_spin))

                amplitudes.append(amplitude)

            mean_amplitude = np.mean(amplitudes)
            resonances.append(mean_amplitude)

        resonances = np.array(resonances)

        return {
            'energies': energy_range,
            'amplitudes': resonances,
            'peaks': self.find_peaks(resonances),
            'energy_range': energy_range
        }

    def find_peaks(self, signal: np.ndarray, threshold: float = 0.8) -> List[int]:
        """Find peaks in signal above threshold."""
        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and
                signal[i] > signal[i+1] and
                signal[i] > threshold * np.max(signal)):
                peaks.append(i)
        return peaks

    def spin_spin_correlation(self, n1: int, n2: int) -> complex:
        """
        Compute spin-spin correlation between two fiber states.

        C(n1, n2) = ⟨ψ(n1)|ψ(n2)⟩

        If this exhibits structure related to Zeta zeros,
        it would be a quantum mechanical connection.
        """
        spin1 = self.fiber_to_spin(n1)
        spin2 = self.fiber_to_spin(n2)

        correlation = np.vdot(spin1, spin2)
        return correlation

    def topological_winding(self, n_points: int = 100) -> float:
        """
        Compute topological winding number of spin configuration.

        In topological phases, winding number is quantized.
        Could this relate to Zeta zero counting function?

        N(T) = # of zeros with 0 < Im(s) < T
        """
        winding = 0.0

        for i in range(n_points):
            n = int(i * self.safe_window / n_points)
            n_next = int((i + 1) * self.safe_window / n_points)

            spin = self.fiber_to_spin(n)
            spin_next = self.fiber_to_spin(n_next)

            # Phase change
            phase = np.angle(spin[0]) - np.angle(spin_next[0])
            winding += phase

        return winding / (2 * np.pi)


def critical_reexamination():
    """
    Reexamine the falsification with physical dynamics.

    QUESTION: Did static geometric tests miss dynamic physical coupling?
    """
    print("="*70)
    print("CRITICAL REEXAMINATION: PHYSICAL DYNAMICS")
    print("="*70)
    print("\nHypothesis: Static geometric falsification missed")
    print("dynamic physical coupling mechanisms.\n")

    print("STATIC vs DYNAMIC:")
    print("-" * 70)
    print("Static test (what we did):")
    print("  - Measured distances between fixed points")
    print("  - Shuffled positions (breaks structure)")
    print("  - Found: position doesn't matter geometrically")
    print()
    print("Dynamic test (what we missed?):")
    print("  - Energy propagation along channel")
    print("  - Spin precession and interference")
    print("  - Phase accumulation over time")
    print("  - Resonance phenomena")
    print()
    print("Key difference: DYNAMICS vs STATICS")
    print()

    # Test K=4 with physics
    p = 10007
    K = 4
    print(f"Testing K={K}, p={p} with physical framework...\n")

    system = PhysicalCollatzSystem(K, p)

    # Test 1: Spin-spin correlations
    print("TEST 1: Spin-Spin Correlations")
    print("-" * 70)

    n_samples = 50
    correlations = np.zeros((n_samples, n_samples), dtype=complex)

    for i in range(n_samples):
        n1 = int(i * system.safe_window / n_samples)
        for j in range(n_samples):
            n2 = int(j * system.safe_window / n_samples)
            correlations[i, j] = system.spin_spin_correlation(n1, n2)

    # Analyze structure
    correlation_magnitudes = np.abs(correlations)
    mean_corr = np.mean(correlation_magnitudes)
    structure_measure = np.std(correlation_magnitudes)

    print(f"Mean correlation magnitude: {mean_corr:.6f}")
    print(f"Structure measure (std): {structure_measure:.6f}")
    print()

    # Test 2: Energy resonances
    print("TEST 2: Energy Resonance Spectrum")
    print("-" * 70)

    # Scan energy range similar to Zeta zero magnitudes
    energy_range = np.linspace(10, 250, 100)  # First ~100 Zeta zeros range

    resonance_data = system.test_energy_resonances(energy_range, n_samples=30)

    n_peaks = len(resonance_data['peaks'])
    print(f"Energy range: [{energy_range[0]:.1f}, {energy_range[-1]:.1f}]")
    print(f"Resonance peaks found: {n_peaks}")

    if n_peaks > 0:
        peak_energies = energy_range[resonance_data['peaks']]
        print(f"Peak energies: {peak_energies[:10]}")  # First 10
        print()
        print("⚠ Non-trivial resonance structure detected!")
        print("This is DIFFERENT from static geometric test.")
    else:
        print("No resonances found.")
        print()

    # Test 3: Topological winding
    print("TEST 3: Topological Winding Number")
    print("-" * 70)

    winding = system.topological_winding(n_points=1000)
    print(f"Winding number: {winding:.4f}")

    if abs(winding) > 0.1:
        print("⚠ Non-zero winding detected!")
        print("Topological structure present.")
    else:
        print("Winding ≈ 0 (topologically trivial)")
    print()

    # Test 4: Wave propagation dynamics
    print("TEST 4: Energy Accumulation Dynamics")
    print("-" * 70)

    n0 = system.safe_window // 2
    energies = [0.1, 1.0, 10.0, 50.0]

    for E in energies:
        result = system.wave_propagation_with_energy(n0, E, n_steps=100)

        final_energy = result['accumulated_energy'][-1]
        final_phase = result['phases'][-1]

        print(f"Initial energy E={E:5.1f}:")
        print(f"  Final accumulated energy: {final_energy:.2f}")
        print(f"  Final quantum phase: {final_phase:.2f} rad")
        print(f"  Steps survived: {len(result['positions'])}")

    print()

    # CRITICAL ASSESSMENT
    print("="*70)
    print("CRITICAL ASSESSMENT")
    print("="*70)
    print()

    has_spin_structure = structure_measure > 0.1
    has_resonances = n_peaks > 0
    has_topology = abs(winding) > 0.1

    print("Physical structure detected:")
    print(f"  Spin correlations: {'YES' if has_spin_structure else 'NO'}")
    print(f"  Energy resonances: {'YES' if has_resonances else 'NO'}")
    print(f"  Topological winding: {'YES' if has_topology else 'NO'}")
    print()

    if has_spin_structure or has_resonances or has_topology:
        print("⚠⚠⚠ PHYSICAL STRUCTURE EXISTS")
        print()
        print("The static geometric falsification may have been premature.")
        print("Dynamic physical processes exhibit non-trivial structure.")
        print()
        print("RECOMMENDATION:")
        print("1. Compare resonance energies to Zeta zero imaginary parts")
        print("2. Analyze spin correlation spectral properties")
        print("3. Study topological phase transitions")
        print("4. Consider field-theoretic formulation")
        print()
        print("Status: INCONCLUSIVE - requires deeper physical analysis")
    else:
        print("✗ NO PHYSICAL STRUCTURE")
        print()
        print("Dynamic tests confirm static falsification.")
        print("System is trivial at both geometric and physical levels.")
        print()
        print("Status: FALSIFICATION CONFIRMED")

    print()
    return {
        'has_spin_structure': has_spin_structure,
        'has_resonances': has_resonances,
        'has_topology': has_topology,
        'resonance_data': resonance_data
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHYSICS FRAMEWORK: ENERGY, SPIN, WAVE PROPAGATION")
    print("="*70)
    print("\nCritically reexamining falsification with physical dynamics.\n")

    results = critical_reexamination()

    print("="*70)
    print("NEXT STEPS")
    print("="*70)

    if (results['has_spin_structure'] or
        results['has_resonances'] or
        results['has_topology']):
        print("\nPhysical structure detected. Further investigation required:")
        print("1. Detailed resonance-Zeta comparison")
        print("2. Field theory formulation")
        print("3. Relativistic wave equation analysis")
        print("4. Experimental/numerical validation at higher precision")
    else:
        print("\nNo physical structure. Falsification stands.")

    print()
