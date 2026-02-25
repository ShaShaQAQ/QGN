# Quantum Geometric Nesting (QGN) Analyzer — Design Document

Date: 2026-02-25
Reference: Han, Herzog-Arbeitman, Bernevig, Kivelson, *Phys. Rev. X* **14**, 041004 (2024)

---

## Goal

Implement a numerical toolkit to analyze the degree of Quantum Geometric Nesting (QGN)
in flat-band tight-binding models, following the formalism of PhysRevX.14.041004.

---

## Architecture

**Approach B**: core Python module + independent Jupyter Notebooks per model.

```
QGN/
├── qgn/
│   ├── __init__.py
│   ├── core.py        # P(k), Π^Q, ω̃₀^Q, nesting matrix N^Q
│   ├── models.py      # SSH, custom TB, TBG Wannier → H(k)
│   └── geometry.py    # band structure, Berry curvature, Chern number,
│                      # quantum distance, Fubini-Study metric
├── notebooks/
│   ├── 01_SSH.ipynb          # SSH chain: verify perfect QGN
│   ├── 02_custom_TB.ipynb    # custom model: parameter sweep
│   └── 03_TBG.ipynb          # TBG Wannier: QGN near magic angle
├── docs/plans/
│   └── 2026-02-25-QGN-design.md
└── README.md
```

Dependencies: `numpy`, `scipy`, `matplotlib`

---

## Core Algorithm (`qgn/core.py`)

### Step 1 — Projection matrix P(k)  [Eq. 4]

From tight-binding diagonalization we obtain all Bloch eigenvectors U(k) of shape
`[Nk, Norb, Norb]`. The user specifies `flat_bands: list[int]` to select which
band indices form the flat-band subspace.

```
P_μν(k) = Σ_{n ∈ flat_bands} U_μn(k) U†_nν(k)
Q_μν(k) = δ_μν - P_μν(k)
```

### Step 2 — Nesting operator Π^Q  [Eq. 5 & 6]

For each wave vector Q on a grid, assemble the Norb² × Norb² Hermitian matrix:

- **p-p channel**:
  Π^{p-p,Q}_{μ'ν';μν} = (1/V) Σ_k [ P*_{μ'μ}(Q/2+k) Q_{νν'}(Q/2-k) + (P↔Q) ]

- **p-h channel**:
  Π^{p-h,Q}_{μ'ν';μν} = (1/V) Σ_k [ P_{μ'μ}(k+Q/2) Q_{νν'}(k-Q/2) + (P↔Q) ]

Implementation: vectorized k-summation using numpy einsum over the k-mesh.

### Step 3 — Nestability ω̃₀^Q

```
ω̃₀^Q = λ_min(Π^Q)
```

Computed for all Q on the BZ grid → heatmap of nestability.
Perfect QGN at Q means ω̃₀^Q = 0.

### Step 4 — Nesting matrix N^Q  (optional, when ω̃₀^Q ≈ 0)

N^Q = null space of Π^Q (via SVD, singular values below threshold).
Reveals the favored order parameter type (SC, DW, ferromagnet, etc.)

---

## Geometry Module (`qgn/geometry.py`)

All quantities derived from P(k) already computed in Step 1.

**Band structure**: eigenvalues of H(k) along high-symmetry path.

**Berry curvature** (gauge-invariant, from projection matrix):
```
Ω_n(k) = -2 Im tr[ P ∂_{kx} P ∂_{ky} P ]
```

**Chern number** (Fukui-Hatsugai-Suzuki lattice method, no gauge fixing needed):
```
C = (1/2π) Σ_{plaquettes} F_12(k)
```

**Quantum distance**:
```
d(k, k') = sqrt( N_flat - tr[P(k) P(k')] )
```

**Fubini-Study metric** (integrated, relates to Cooper pair mass):
```
g = ∫ d²k/(2π)² Tr(∂_i P ∂_j P) / 2
```

---

## Models (`qgn/models.py`)

All models expose: `get_hamiltonian(k) -> H[Norb, Norb]`

### Model 1: SSH Chain
- Parameters: `t1`, `t2`; 2 orbitals per unit cell; 1D BZ k ∈ [-π, π)
- Perfect QGN expected when |t1| = |t2|

### Model 2: Custom Tight-Binding
- User provides hopping dict `{R: t_matrix}`, lattice vectors `a1, a2`, `flat_bands`
- Reuses summation logic from existing `HF_calc_moire/moire/TBmodel.py`

### Model 3: TBG Wannier Low-Energy Model
- 4-band model (2 flat bands per valley/spin) with Koshino 2018 Wannier parameters
  hardcoded at magic angle θ ≈ 1.05°
- User can override parameters for off-magic-angle studies
- Perfect QGN guaranteed by time-reversal + chiral symmetry (Sec. VI.B of paper)

---

## Notebook Structure (each notebook)

1. Model parameter definition
2. Band structure plot + Berry curvature heatmap + Chern number
3. Flat band index confirmation + P(k) matrix element visualization
4. ω̃₀^Q heatmap over BZ (p-p and p-h channels separately)
5. Nesting matrix N^Q analysis at Q with minimum ω̃₀^Q
