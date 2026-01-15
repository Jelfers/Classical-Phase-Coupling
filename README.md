# Classical Phase Coupling: Collatz-Zeta Verification

**Mathematical analysis of a skew-product Collatz dynamical system coupled with modular arithmetic.**

## Overview

This repository contains a rigorous verification suite for analyzing arithmetic carry mechanisms in an extended Collatz dynamical system. The work examines how discrete carry terms can create exact invariant structures when coupling parameters align with Collatz cycle geometry.

**No metaphysics. No infinities. Only arithmetic gates.**

## Mathematical System

### System Definition

We define a skew-product dynamical system:

```
T : ℤ × ℤ_p → ℤ × ℤ_p
```

Where:
- Base space: Integer Collatz dynamics (w ∈ ℤ)
- Fiber space: Modular arithmetic (n ∈ ℤ_p)
- Coupling: Integer carry from modular overflow

### Evolution Rules

**Even step** (w even):
```
(w, n) ↦ (w/2, n/2 mod p)
```

**Odd step** (w odd):
```
(w, n) ↦ (3w + 1 + c, Kn mod p)
```

where the **carry term** is:
```
c = ⌊Kn/p⌋
```

This carry term is the **only coupling** between base and fiber dynamics.

## Key Mathematical Structures

### 1. The Collatz 3-Cycle as Gate

The classical Collatz cycle:
```
C = {1 → 4 → 2 → 1}
```

The fiber can only inject information into the base when `w = 1` (odd step entry).

**Carry Gate Condition:** Remaining on the cycle requires `c = 0` at `w = 1`.

### 2. Safe Window

For carry-free evolution at `w = 1`:
```
c = 0  ⟺  Kn < p
```

This defines the **safe window**:
```
W_K = [0, (p-1)/K]
```

Points outside this window generate carry and are ejected from the cycle.

### 3. Fiber Return Map

For a trajectory that completes one full Collatz cycle `{1→4→2→1}`, the fiber evolution is:

- **1 → 4**: `n ↦ Kn`
- **4 → 2**: `n ↦ n/2`
- **2 → 1**: `n ↦ n/2`

Composing these gives the **return map**:
```
R_K(n) ≡ (K/4)n mod p
```

### 4. The K=4 Invariance

When `K = 4`, the return map becomes:
```
R_4(n) ≡ n mod p
```

This is the **identity map**. Consequences:

- Any `n ∈ W_4` returns to itself after one full Collatz cycle
- No modular wraparound occurs
- The carry gate remains closed indefinitely
- **Exact invariance** (not asymptotic)

This creates a nontrivial invariant set:
```
I_4 = {(1, n) : n ∈ W_4}
```

### 5. K≠4 Extinction

For `K ≠ 4`, the return map is multiplicative with ratio `K/4`:

- **K < 4**: Contraction → eventual decay to small values → extinction
- **K > 4**: Expansion → modular wraparound → carry generation → ejection

This creates a **multiplicative sieve** that filters out almost all initial conditions.

## Mathematical Claims

This work establishes the following:

### Verified Claims

1. **Carry Gate Arithmetic**: `c = 0 ⟺ Kn < p` (exact condition)
2. **Return Map Formula**: `R_K(n) ≡ (K/4)n mod p` (derived and verified)
3. **K=4 Exact Invariance**: `R_4(n) = n` for all `n ∈ ℤ_p` (identity map)
4. **Safe Window Structure**: `|W_K| = ⌊(p-1)/K⌋ + 1` (window size formula)

### Conjectures Under Test

1. **K=4 Uniqueness**: K=4 is the only coupling constant that produces exact invariance
2. **K≠4 Rapid Extinction**: Non-K=4 couplings lead to rapid trajectory ejection
3. **Spectral Structure**: Survivor statistics may exhibit properties similar to arithmetic spectral systems

## Verification Suite

### Installation

Requires Python 3.9+ with NumPy:

```bash
pip install numpy
```

### Running Tests

Basic run with default parameters (p=1009, 100 samples):
```bash
python verification_suite.py
```

Custom prime and sample size:
```bash
python verification_suite.py <prime> <n_samples>

# Example: Use p=10007 with 200 samples
python verification_suite.py 10007 200
```

### Test Suite Components

The verification suite implements five rigorous tests:

#### TEST 1: Carry Gate Condition
Verifies the equivalence `c = 0 ⟺ Kn < p` across the full modular range.

#### TEST 2: Return Map Formula
Confirms that the computed return map matches the theoretical formula `R_K(n) ≡ (K/4)n mod p`.

#### TEST 3: K=4 Exact Invariance
- Verifies `R_4(n) = n` (identity check)
- Tests trajectory survival for points in safe window
- **Critical test**: Falsification of this invalidates the central claim

#### TEST 4: K≠4 Extinction
Tests multiple K values (2, 3, 5, 6, 8, 12) to verify rapid ejection from the Collatz cycle.

#### TEST 5: Return Map Structure
Analyzes orbit lengths under iterated return map to observe multiplicative sieve behavior.

### Interpretation of Results

**PASS Criteria:**
- TEST 1-2: All samples match theoretical predictions (arithmetic validity)
- TEST 3: 100% survival rate for K=4 in safe window (exact invariance)
- TEST 4: <10% survival rate for all K≠4 (sieve effectiveness)
- TEST 5: K=4 shows persistent orbits, K≠4 shows rapid exit

**Falsification:**
- If TEST 3 fails: K=4 invariance claim is false
- If TEST 4 fails: K≠4 extinction is weaker than claimed
- If TEST 1-2 fail: Implementation error (check arithmetic)

## LaTeX Documentation

A comprehensive LaTeX document is provided:

```bash
pdflatex collatz_zeta_foundations.tex
bibtex collatz_zeta_foundations
pdflatex collatz_zeta_foundations.tex
pdflatex collatz_zeta_foundations.tex
```

The document contains:
- Formal system definitions
- Proofs of arithmetic identities
- Analysis of K=4 invariance mechanism
- Discussion of multiplicative sieve structure
- Numerical verification results

## Extended Zeta Coupling Analysis

### Overview

Beyond basic verification, we conducted rigorous investigation of potential connections to Riemann Zeta zero dynamics.

**⚠️ IMPORTANT**: See `CORRECTED_ASSESSMENT.md` for complete and accurate analysis of what was falsified vs what remains valid.

**Quick Summary**:
- ✗ **Direct spectral coupling**: FALSIFIED (position correlation is geometric artifact)
- ✓ **Framework connections**: VALID (both governed by representation theory)
- ? **Dynamic field theory**: UNTESTED (requires theoretical work)

For initial findings, see `ZETA_COUPLING_FINDINGS.md`. For falsification details, see `FINAL_VERDICT.md`.

### Running Extended Analysis

```bash
python zeta_coupling_analysis.py 10007
```

For larger primes (warning: longer runtime):
```bash
python zeta_coupling_analysis.py 100003
```

### Key Findings

**UPDATE: ZETA COUPLING FALSIFIED VIA CONTROL TESTS**

Initial investigation showed promising correlations (p < 0.0001), but maximum-scale testing with full control battery revealed these are **geometric artifacts**:

1. **Shuffled Control Test** (CRITICAL): ✗ FAILED
   - Randomly shuffling survivor positions produces **identical** distance statistics
   - p-value = 1.0 (no difference between true and shuffled positions)
   - **Conclusion**: Position doesn't matter, only density (25% coverage)

2. **Density Artifact**:
   - K=4 safe window contains 25% of all residues mod p
   - High density guarantees proximity to any target sequence
   - Nothing specific to Zeta zeros

3. **Uniform Grid Artifact**:
   - Survivors form regular integer progression {0,1,2,3,...}
   - Perfect spacing (gap = 1) creates maximal coverage
   - Geometric optimization, not number-theoretic structure

4. **Large Prime Scaling**: All primes (10K-1M) show "correlation"
   - But shuffled control fails at all scales
   - Geometric artifacts scale consistently

### Critical Assessment

**Status**: ~~"Suggestive evidence"~~ → **FALSIFIED** → Restored to "motivational analogy only"

**What the control tests revealed**:
- ✗ GUE statistics: Artifact of uniform integer spacing, not quantum chaos
- ✗ Direct correlation: Due to 25% density + uniform grid, not Zeta structure
- ✗ Statistical significance real, but causally meaningless (geometric, not structural)
- ✓ K=4 exact invariance: Still valid (proven mathematically, independent of Zeta)

**The Smoking Gun**: When survivor positions are shuffled (keeping same density), correlation **persists identically**. This definitively proves the "correlation" is geometric, not positional.

**Lesson Learned**: p-values < 0.0001 can be statistically significant yet causally meaningless. Control tests are essential.

See `FINAL_VERDICT.md` for complete falsification analysis.

## Repository Structure

```
.
├── README.md                           # This file
├── CORRECTED_ASSESSMENT.md             # ⭐ Corrected understanding (read this first!)
├── verification_suite.py               # Basic verification tests (K=4 invariance: PASS)
├── group_theoretic_foundations.py      # Rigorous group-theoretic framework
├── deeper_connection_analysis.py       # Framework connections analysis
├── zeta_coupling_analysis.py           # Initial Zeta investigation (showed correlation)
├── maximum_scale_analysis.py           # Control test battery (falsified position correlation)
├── final_diagnostic_test.py            # Uniform grid artifact demonstration
├── quantum_analysis.py                 # Quantum transfer operator analysis
├── physics_framework.py                # Dynamic energy/spin analysis
├── resonance_zeta_comparison.py        # Energy resonance comparison
├── ZETA_COUPLING_FINDINGS.md           # Initial findings (pre-falsification)
├── FINAL_VERDICT.md                    # Position correlation falsification
├── RECONSIDERING_FRAMEWORK.md          # Static vs dynamic distinction
├── collatz_zeta_foundations.tex        # LaTeX mathematical documentation
└── collatz_zeta_foundations.pdf        # Compiled PDF (manual commit)
```

## Interpretation Guidelines

### What This Work Establishes

This work demonstrates:
- A valid extension of Collatz dynamics via skew-product structure
- Exact arithmetic invariance at K=4 through carry cancellation
- Multiplicative sieve behavior for K≠4
- A discrete control gate mechanism (not continuous dynamics)

### What This Work Does NOT Claim

- ❌ Direct spectral coupling to Riemann Zeta zeros (falsified via control tests)
- ❌ Proof of Collatz conjecture
- ❌ Continuous dynamical interpretation
- ❌ Force-based or energy-based mechanisms

### What This Work DOES Establish Beyond K=4 Invariance

- ✅ Valid framework connections through representation theory
- ✅ Group-theoretic classification of return map dynamics
- ✅ Complementary relationship (identity vs non-identity principles)
- ✅ Symmetry → conservation correspondence (Noether-like)

### Relationship to Zeta Zeros

**⚠️ CORRECTED ASSESSMENT** (see `CORRECTED_ASSESSMENT.md` for full details)

**What we tested**:
- ✓ Large primes (up to 1,000,003)
- ✓ 100+ verified Zeta zeros
- ✓ Full control battery (shuffled, sparse, random)
- ✓ Alternative K values

**What we found**:

**✗ FALSIFIED - Direct Spectral Coupling:**
- **Shuffled control FAILED** (smoking gun): Position irrelevant, only density matters
- GUE statistics: Artifact of uniform integer spacing
- Position correlation: Geometric artifact from 25% density + uniform grid {0,1,2,3,...}
- Statistical significance (p < 0.0001) was real but causally meaningless

**✓ VALID - Framework Connections:**
- **Representation theory**: Both K=4 and Zeta systems governed by same framework
- **Identity principle**: R₄ = I (identity element in (ℤ_p)*) parallels identity in quantum mechanics
- **Symmetry → conservation**: Both exhibit Noether-like correspondence
- **Complementary extremes**: K=4 (trivial rep) and Zeta (non-trivial rep) as opposite ends of same spectrum

**? UNTESTED - Dynamic Field Theory:**
- Energy spectrum from proper Hamiltonian formulation
- Wave propagation and relativistic constraints
- See `RECONSIDERING_FRAMEWORK.md` for theoretical framework

**The "0 and 1" Analogy:**

K=4 and Zeta are like 0 and 1 in mathematics:
- ✓ Same framework (representation theory)
- ✓ Opposite extremes (identity vs non-identity)
- ✓ Governed by same principles (group actions)
- ✗ Don't directly equal or correlate

**Current status**: Direct coupling falsified, but valid framework connections through representation theory remain. Both systems are examples of group-theoretic principles.

See `CORRECTED_ASSESSMENT.md` for complete corrected analysis.

## Technical Notes

### Prime Selection

- Use primes `p > 4K` to ensure safe window exists
- Larger primes give better resolution but slower computation
- Recommended range: `p ∈ [1009, 10007]` for testing
- Use `p > 100000` for publication-quality verification

### Numerical Precision

- All arithmetic is exact (integer and modular)
- No floating-point error accumulation
- Carry terms are computed via integer division
- Modular inverse computed via extended Euclidean algorithm

### Computational Complexity

- Single trajectory: O(steps)
- Return map: O(log p) per iteration (modular arithmetic)
- Full verification suite: O(n_samples × max_steps)
- Typical runtime: 10-60 seconds for default parameters

## Future Directions

### Mathematical Extensions

1. **Formal proof of K=4 uniqueness**: Show no other K produces exact invariance
2. **Measure-theoretic analysis**: Compute invariant measure on safe window
3. **Generalization to other cycles**: Extend beyond {1,4,2} cycle
4. **Higher-dimensional fibers**: ℤ_p^n instead of ℤ_p

### Computational Investigations

1. **Statistical tests**: Compare survivor distributions to spectral statistics
2. **Larger primes**: Verify behavior at p ~ 10^6 to 10^9
3. **K-space mapping**: Dense sampling of K parameter space
4. **Cycle detection**: Identify all periodic orbits under return map

### Theoretical Questions

1. Does the safe window measure vanish as p → ∞?
2. What is the distribution of ejection times for K≠4?
3. Can this mechanism be related to other number-theoretic systems?
4. Is there a deeper connection to L-function arithmetic?

## Citation

If you use this work, please cite:

```
[Author]. "Classical Phase Coupling: Arithmetic Invariance in
Skew-Product Collatz Dynamics." (2026).
```

## License

[To be determined]

## Contact

[To be determined]

---

**Truth over comfort. Rigorous mathematical approach. Number Theory meets Physics minus interpretation.**