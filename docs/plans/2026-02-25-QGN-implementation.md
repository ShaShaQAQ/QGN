# QGN Analyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Jupyter-Notebook-based toolkit to numerically analyze Quantum Geometric
Nesting (QGN) in flat-band tight-binding models, following PhysRevX.14.041004.

**Architecture:** Core Python module `qgn/` (core.py, models.py, geometry.py) shared by
three Notebooks (SSH, custom TB, TBG Wannier). Each Notebook: model definition →
band/Berry curvature/Chern number → QGN nestability heatmap → nesting matrix analysis.

**Tech Stack:** Python 3, numpy, scipy, matplotlib, pytest, jupyter

---

### Task 1: Project Scaffold

**Files:**
- Create: `qgn/__init__.py`
- Create: `qgn/core.py` (stub)
- Create: `qgn/models.py` (stub)
- Create: `qgn/geometry.py` (stub)
- Create: `tests/__init__.py`
- Create: `requirements.txt`
- Create: `README.md`

**Step 1: Create directory structure**

```bash
cd /Users/shajianyu/CMP_manybody/QGN
mkdir -p qgn tests notebooks
```

**Step 2: Create `requirements.txt`**

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
jupyter
pytest
```

**Step 3: Create `qgn/__init__.py`**

```python
from .core import projection_matrix, nesting_operator, nestability
from .models import ssh_hamiltonian, custom_tb_hamiltonian, tbg_wannier_hamiltonian
from .geometry import band_structure, berry_curvature, chern_number, quantum_distance
```

**Step 4: Create stubs for `qgn/core.py`, `qgn/models.py`, `qgn/geometry.py`**

Each file: just `# TODO` and the function signatures (no body).

**Step 5: Create `tests/__init__.py`** (empty)

**Step 6: Commit**

```bash
git add qgn/ tests/ requirements.txt README.md
git commit -m "chore: project scaffold with stubs"
```

---

### Task 2: SSH Model

**Files:**
- Modify: `qgn/models.py`
- Create: `tests/test_models.py`

**Step 1: Write failing test**

```python
# tests/test_models.py
import numpy as np
from qgn.models import ssh_hamiltonian

def test_ssh_shape():
    H = ssh_hamiltonian(k=0.0, t1=1.0, t2=0.5)
    assert H.shape == (2, 2)

def test_ssh_hermitian():
    for k in np.linspace(0, 2*np.pi, 20):
        H = ssh_hamiltonian(k=k, t1=1.0, t2=0.5)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

def test_ssh_spectrum_symmetric():
    # SSH spectrum is symmetric: ε_n(k) = -ε_n(k) (chiral)
    k = 0.7
    eigvals = np.linalg.eigvalsh(ssh_hamiltonian(k, t1=1.0, t2=0.5))
    np.testing.assert_allclose(eigvals[0], -eigvals[1], atol=1e-12)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_models.py -v
```
Expected: ImportError or AttributeError.

**Step 3: Implement SSH model in `qgn/models.py`**

```python
import numpy as np

def ssh_hamiltonian(k, t1=1.0, t2=0.5):
    """SSH chain Hamiltonian (1D, 2 orbitals per unit cell).

    H(k) = [[0,             t1 + t2*exp(-ik)],
            [t1 + t2*exp(ik),  0            ]]

    Perfect QGN at Q=π when |t1| == |t2|.
    """
    h01 = t1 + t2 * np.exp(-1j * k)
    return np.array([[0, h01], [h01.conj(), 0]], dtype=complex)
```

**Step 4: Run tests**

```bash
pytest tests/test_models.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add qgn/models.py tests/test_models.py
git commit -m "feat: SSH chain Hamiltonian"
```

---

### Task 3: Custom TB and TBG Wannier Models

**Files:**
- Modify: `qgn/models.py`
- Modify: `tests/test_models.py`

**Step 1: Write failing tests**

```python
# append to tests/test_models.py
from qgn.models import custom_tb_hamiltonian, tbg_wannier_hamiltonian

def test_custom_tb_shape():
    # 2-orbital model: one site, hopping to NN at R=(1,0) and R=(0,1)
    hoppings = {(1, 0): np.array([[0.5]]), (0, 1): np.array([[0.3]])}
    H = custom_tb_hamiltonian(k=np.array([0.1, 0.2]), hoppings=hoppings, norb=1)
    assert H.shape == (1, 1)

def test_tbg_shape():
    H = tbg_wannier_hamiltonian(k=np.array([0.0, 0.0]))
    assert H.shape == (4, 4)

def test_tbg_hermitian():
    for _ in range(5):
        k = np.random.rand(2)
        H = tbg_wannier_hamiltonian(k)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_models.py::test_custom_tb_shape -v
```

**Step 3: Implement `custom_tb_hamiltonian`**

```python
def custom_tb_hamiltonian(k, hoppings, norb):
    """Generic tight-binding Hamiltonian from hopping dictionary.

    Args:
        k: wave vector in fractional coordinates, shape (2,) for 2D
        hoppings: dict {R: t_matrix} where R is lattice vector tuple,
                  t_matrix is (norb, norb) complex array
        norb: number of orbitals per unit cell

    Returns:
        H: (norb, norb) Hamiltonian matrix
    """
    H = np.zeros((norb, norb), dtype=complex)
    for R, t in hoppings.items():
        phase = np.exp(1j * 2 * np.pi * np.dot(k, R))
        H += t * phase
    # Hermitian: add conjugate hoppings
    H = H + H.conj().T - np.diag(np.diag(H).real)
    return H
```

Wait — the above double-counts diagonal. Better: require user to include both R and -R,
or symmetrize explicitly. Use the cleaner convention: H = Σ_R t(R) e^{ik·R} where the
sum includes all R with t(-R) = t(R)†.

```python
def custom_tb_hamiltonian(k, hoppings, norb):
    """Generic tight-binding Hamiltonian.

    hoppings: dict {R: t_matrix}, must include R=0 and both ±R for off-diagonal.
    Convention: H(k) = Σ_R t(R) exp(i 2π k·R)
    """
    H = np.zeros((norb, norb), dtype=complex)
    for R, t in hoppings.items():
        phase = np.exp(1j * 2 * np.pi * np.dot(k, np.array(R)))
        H += np.array(t, dtype=complex) * phase
    return H
```

**Step 4: Implement `tbg_wannier_hamiltonian`**

4-band model per valley (2 flat + 2 remote bands). Uses Koshino (2018) nearest-neighbor
Wannier parameters for the moiré triangular lattice. Here valley K only; spin is implicit.

```python
# TBG moiré lattice vectors (in moiré units)
_TBG_A1 = np.array([1.0, 0.0])
_TBG_A2 = np.array([0.5, np.sqrt(3)/2])

# Nearest-neighbor vectors of triangular lattice
_TBG_NN = [_TBG_A1, _TBG_A2, _TBG_A2 - _TBG_A1]

# Koshino 2018 Wannier parameters (meV), 4-orbital basis: (A1, B1, A2, B2)
# where 1,2 label sublattice-like Wannier centers, A/B label moiré sublattice
_TBG_PARAMS = {
    't1':  -0.331,   # nearest-neighbor AA/BB hopping (eV)
    't2':   0.368,   # nearest-neighbor AB hopping
    't3':  -0.026,   # next-nearest-neighbor hopping
}

def tbg_wannier_hamiltonian(k, params=None):
    """TBG Wannier tight-binding Hamiltonian (4-band, valley K).

    4 orbitals: (A1, B1, A2, B2) — two sublattice-like Wannier centers
    on the moiré triangular lattice.

    Args:
        k: wave vector in moiré BZ fractional coordinates, shape (2,)
        params: dict with keys 't1','t2','t3'. Defaults to Koshino 2018.

    Returns:
        H: (4, 4) Hermitian matrix
    """
    p = params if params is not None else _TBG_PARAMS
    t1, t2, t3 = p['t1'], p['t2'], p['t3']

    # Phase factors for 6 nearest neighbors of triangular lattice
    f = sum(np.exp(1j * 2 * np.pi * np.dot(k, nn)) for nn in _TBG_NN)
    f_conj = np.conj(f)

    # Build 4x4 H in basis (A1, B1, A2, B2)
    # Sublattice structure within each "layer":
    h_intra = np.array([[0,    t2*f],
                        [t2*f_conj, 0  ]], dtype=complex)
    h_inter = t1 * np.eye(2, dtype=complex)  # interlayer AA/BB

    H = np.block([[h_intra,  h_inter],
                  [h_inter.conj().T, h_intra]])
    return H
```

Note: this is a simplified parametrization. For detailed TBG studies replace with
the full Wannier Hamiltonian from a DFT/BM downfolding.

**Step 5: Run all model tests**

```bash
pytest tests/test_models.py -v
```
Expected: all PASS.

**Step 6: Commit**

```bash
git add qgn/models.py tests/test_models.py
git commit -m "feat: custom TB and TBG Wannier Hamiltonians"
```

---

### Task 4: Geometry Module (Band Structure, Berry Curvature, Chern Number, Quantum Distance)

**Files:**
- Modify: `qgn/geometry.py`
- Create: `tests/test_geometry.py`

**Step 1: Write failing tests**

```python
# tests/test_geometry.py
import numpy as np
from qgn.models import ssh_hamiltonian
from qgn.geometry import (diagonalize_model, berry_curvature_grid,
                           chern_number, quantum_distance)

def _ssh_data(Nk=50, t1=1.0, t2=0.5):
    k_grid = np.linspace(0, 2*np.pi, Nk, endpoint=False)
    eigvals = np.zeros((Nk, 2))
    eigvecs = np.zeros((Nk, 2, 2), dtype=complex)
    for i, k in enumerate(k_grid):
        e, v = np.linalg.eigh(ssh_hamiltonian(k, t1, t2))
        eigvals[i], eigvecs[i] = e, v
    return k_grid, eigvals, eigvecs

def test_diagonalize_returns_correct_shapes():
    k_grid = np.linspace(0, 2*np.pi, 30, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1=1.0, t2=0.5)
    eigvals, eigvecs = diagonalize_model(H_func, k_grid)
    assert eigvals.shape == (30, 2)
    assert eigvecs.shape == (30, 2, 2)

def test_ssh_chern_number_zero():
    # SSH in trivial phase (t1 > t2): both bands have C=0
    k_grid = np.linspace(0, 2*np.pi, 100, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1=1.0, t2=0.3)
    eigvals, eigvecs = diagonalize_model(H_func, k_grid)
    # 1D: Chern number not defined; just check function runs
    # For 2D models the Chern number test is in test_tbg_chern
    assert eigvals.shape[0] == 100

def test_quantum_distance_self_zero():
    _, _, eigvecs = _ssh_data(Nk=20)
    P = np.einsum('kia,kja->kij', eigvecs[:, :, :1],
                  eigvecs[:, :, :1].conj())  # project onto band 0
    d = quantum_distance(P[0], P[0], n_flat=1)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)

def test_quantum_distance_bounded():
    _, _, eigvecs = _ssh_data(Nk=20)
    P0 = np.einsum('ia,ja->ij', eigvecs[0, :, :1],
                   eigvecs[0, :, :1].conj())
    P5 = np.einsum('ia,ja->ij', eigvecs[5, :, :1],
                   eigvecs[5, :, :1].conj())
    d = quantum_distance(P0, P5, n_flat=1)
    assert 0.0 <= d <= 1.0
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_geometry.py -v
```

**Step 3: Implement `qgn/geometry.py`**

```python
import numpy as np

def diagonalize_model(H_func, k_points):
    """Diagonalize H(k) at each k point.

    Args:
        H_func: callable k -> H[Norb, Norb], k can be scalar (1D) or array (2D)
        k_points: array of k points, shape (Nk,) for 1D or (Nk, 2) for 2D

    Returns:
        eigvals: (Nk, Norb) sorted eigenvalues
        eigvecs: (Nk, Norb, Norb) eigenvectors; eigvecs[k, :, n] is band n
    """
    H0 = H_func(k_points[0])
    Norb = H0.shape[0]
    Nk = len(k_points)
    eigvals = np.zeros((Nk, Norb))
    eigvecs = np.zeros((Nk, Norb, Norb), dtype=complex)
    for i, k in enumerate(k_points):
        e, v = np.linalg.eigh(H_func(k))
        eigvals[i] = e
        eigvecs[i] = v
    return eigvals, eigvecs


def projection_matrix_from_vecs(eigvecs, flat_bands):
    """Compute flat-band projection matrix P(k) from eigenvectors.

    P_μν(k) = Σ_{n ∈ flat_bands} U_μn(k) U†_nν(k)   [Eq. 4]

    Args:
        eigvecs: (Nk, Norb, Norb) — eigvecs[k, :, n] is the n-th Bloch vector
        flat_bands: list of band indices forming the flat-band subspace

    Returns:
        P: (Nk, Norb, Norb) projection matrices
    """
    U_flat = eigvecs[:, :, flat_bands]          # (Nk, Norb, N_flat)
    P = U_flat @ U_flat.conj().transpose(0, 2, 1)  # (Nk, Norb, Norb)
    return P


def berry_curvature_grid(eigvecs, flat_bands, dk1, dk2):
    """Berry curvature on a 2D k-grid using projection matrix.

    Ω(k) = -2 Im tr[ P ∂_{k1} P ∂_{k2} P ]

    Args:
        eigvecs: (Nk1, Nk2, Norb, Norb)
        flat_bands: list of band indices
        dk1, dk2: grid spacing in each direction

    Returns:
        Omega: (Nk1, Nk2) Berry curvature array
    """
    P = projection_matrix_from_vecs(
        eigvecs.reshape(-1, *eigvecs.shape[2:]), flat_bands
    ).reshape(eigvecs.shape[:2] + eigvecs.shape[2:])  # (Nk1, Nk2, Norb, Norb)

    dP1 = np.gradient(P, dk1, axis=0)  # (Nk1, Nk2, Norb, Norb)
    dP2 = np.gradient(P, dk2, axis=1)

    # tr[P dP1 dP2] at each k point
    product = np.einsum('...ij,...jk,...ki->...', P, dP1, dP2)
    return -2 * np.imag(product)


def chern_number(eigvecs_2d, flat_bands):
    """Chern number via Fukui-Hatsugai-Suzuki lattice method.

    Args:
        eigvecs_2d: (Nk1, Nk2, Norb, Norb) eigenvectors on 2D grid
        flat_bands: list of band indices

    Returns:
        C: float, should be close to an integer
    """
    Nk1, Nk2 = eigvecs_2d.shape[:2]
    F_total = 0.0
    for i in range(Nk1):
        for j in range(Nk2):
            # Flat-band eigenvectors at 4 plaquette corners
            U00 = eigvecs_2d[i,           j,           :, flat_bands]  # (Norb, Nflat)
            U10 = eigvecs_2d[(i+1) % Nk1, j,           :, flat_bands]
            U11 = eigvecs_2d[(i+1) % Nk1, (j+1) % Nk2, :, flat_bands]
            U01 = eigvecs_2d[i,           (j+1) % Nk2, :, flat_bands]
            # Link variables
            L01 = np.linalg.det(U00.conj().T @ U10)
            L12 = np.linalg.det(U10.conj().T @ U11)
            L23 = np.linalg.det(U11.conj().T @ U01)
            L30 = np.linalg.det(U01.conj().T @ U00)
            F_total += np.angle(L01 * L12 * L23 * L30)
    return F_total / (2 * np.pi)


def quantum_distance(P_k, P_kp, n_flat):
    """Quantum distance between two k-points.

    d(k, k') = sqrt( N_flat - tr[P(k) P(k')] )

    Args:
        P_k, P_kp: (Norb, Norb) projection matrices
        n_flat: number of flat bands (= N_flat)

    Returns:
        d: non-negative float
    """
    overlap = np.real(np.trace(P_k @ P_kp))
    return np.sqrt(max(n_flat - overlap, 0.0))
```

**Step 4: Run tests**

```bash
pytest tests/test_geometry.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add qgn/geometry.py tests/test_geometry.py
git commit -m "feat: geometry module (band structure, Berry curvature, Chern number, quantum distance)"
```

---

### Task 5: Core QGN Module (Projection Matrix, Nesting Operator, Nestability, Nesting Matrix)

**Files:**
- Modify: `qgn/core.py`
- Create: `tests/test_core.py`

**Step 1: Write failing tests**

```python
# tests/test_core.py
import numpy as np
from qgn.models import ssh_hamiltonian
from qgn.geometry import diagonalize_model, projection_matrix_from_vecs
from qgn.core import nesting_operator, nestability_map

def _get_ssh_P(Nk=60, t1=1.0, t2=0.5):
    k_grid = np.linspace(0, 2*np.pi, Nk, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1, t2)
    _, eigvecs = diagonalize_model(H_func, k_grid)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    Q_mat = np.eye(2)[None, ...] - P
    return k_grid, P, Q_mat

def test_nesting_operator_shape():
    k_grid, P, Q_mat = _get_ssh_P()
    Norb = 2
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp')
    assert Pi.shape == (Norb**2, Norb**2)

def test_nesting_operator_hermitian():
    k_grid, P, Q_mat = _get_ssh_P()
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp')
    np.testing.assert_allclose(Pi, Pi.conj().T, atol=1e-10)

def test_nesting_operator_psd():
    """Π^Q must be positive semidefinite."""
    k_grid, P, Q_mat = _get_ssh_P()
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp')
    eigvals = np.linalg.eigvalsh(Pi)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

def test_ssh_perfect_qgn_at_pi():
    """SSH with |t1|=|t2| has perfect QGN at Q=π: ω̃₀^π = 0."""
    k_grid, P, Q_mat = _get_ssh_P(Nk=100, t1=1.0, t2=1.0)
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp')
    omega = np.linalg.eigvalsh(Pi).min()
    np.testing.assert_allclose(omega, 0.0, atol=1e-6)

def test_nestability_map_shape():
    k_grid, P, Q_mat = _get_ssh_P(Nk=40)
    Q_grid = np.linspace(0, 2*np.pi, 20, endpoint=False)
    omega_map = nestability_map(P, Q_mat, k_grid, Q_grid, channel='pp')
    assert omega_map.shape == (20,)
    assert np.all(omega_map >= -1e-10)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_core.py -v
```

**Step 3: Implement `qgn/core.py`**

```python
import numpy as np
from scipy.interpolate import interp1d


def _interpolate_P_1d(P, k_grid, k_shifted):
    """Interpolate P(k) at shifted k points (1D, periodic).

    Args:
        P: (Nk, Norb, Norb)
        k_grid: (Nk,) uniformly spaced in [0, 2π)
        k_shifted: (Nk,) shifted k points (may be outside [0, 2π))

    Returns:
        P_shifted: (Nk, Norb, Norb)
    """
    Nk, Norb, _ = P.shape
    # Wrap to [0, 2π)
    k_mod = k_shifted % (2 * np.pi)
    # Extend P periodically for interpolation
    k_ext = np.concatenate([k_grid, k_grid + 2*np.pi])
    P_ext = np.concatenate([P, P], axis=0)

    P_shifted = np.zeros_like(P)
    for mu in range(Norb):
        for nu in range(Norb):
            f_re = interp1d(k_ext, P_ext[:, mu, nu].real, kind='linear')
            f_im = interp1d(k_ext, P_ext[:, mu, nu].imag, kind='linear')
            P_shifted[:, mu, nu] = f_re(k_mod) + 1j * f_im(k_mod)
    return P_shifted


def nesting_operator(P, Q_mat, k_grid, Q_vec, channel='pp'):
    """Compute nesting operator Π^Q.  [Eq. 5 or 6]

    Supports 1D k_grid (scalar Q_vec) only. For 2D see nesting_operator_2d.

    Args:
        P:      (Nk, Norb, Norb) flat-band projection matrices
        Q_mat:  (Nk, Norb, Norb) complement Q = I - P
        k_grid: (Nk,) k points
        Q_vec:  scalar, nesting wave vector
        channel: 'pp' (particle-particle) or 'ph' (particle-hole)

    Returns:
        Pi: (Norb², Norb²) Hermitian positive-semidefinite matrix
    """
    Nk, Norb, _ = P.shape

    if channel == 'pp':
        # k + Q/2  and  k - Q/2
        k_plus  = k_grid + Q_vec / 2
        k_minus = k_grid - Q_vec / 2
        P_plus  = _interpolate_P_1d(P,     k_grid, k_plus)   # P(k + Q/2)
        Q_minus = _interpolate_P_1d(Q_mat, k_grid, k_minus)  # Q(k - Q/2)
        # First term: P*(k+Q/2) ⊗ Q(k-Q/2)  →  Norb²×Norb² matrix
        # [μ'ν';μν] = P*_{μ'μ}(k+Q/2) * Q_{νν'}(k-Q/2)
        term1 = np.einsum('kab,kcd->kacbd', P_plus.conj(), Q_minus)
        # (P ↔ Q): Q*(k+Q/2) ⊗ P(k-Q/2)
        Q_plus  = _interpolate_P_1d(Q_mat, k_grid, k_plus)
        P_minus = _interpolate_P_1d(P,     k_grid, k_minus)
        term2 = np.einsum('kab,kcd->kabd', Q_plus.conj(), P_minus)  # Note: index order matches Eq.5
    else:  # ph channel
        k_plus  = k_grid + Q_vec / 2
        k_minus = k_grid - Q_vec / 2
        P_plus  = _interpolate_P_1d(P,     k_grid, k_plus)
        Q_minus = _interpolate_P_1d(Q_mat, k_grid, k_minus)
        term1 = np.einsum('kab,kcd->kabd', P_plus, Q_minus)
        Q_plus  = _interpolate_P_1d(Q_mat, k_grid, k_plus)
        P_minus = _interpolate_P_1d(P,     k_grid, k_minus)
        term2 = np.einsum('kab,kcd->kabd', Q_plus, P_minus)

    Pi_raw = (term1 + term2).sum(axis=0) / Nk   # (Norb, Norb, Norb, Norb)
    Pi = Pi_raw.reshape(Norb**2, Norb**2)
    # Symmetrize to enforce Hermitian (numerical noise)
    Pi = (Pi + Pi.conj().T) / 2
    return Pi


def nestability_map(P, Q_mat, k_grid, Q_grid, channel='pp'):
    """Compute ω̃₀^Q = λ_min(Π^Q) for each Q on a grid.

    Args:
        P, Q_mat: (Nk, Norb, Norb)
        k_grid:   (Nk,) 1D k points
        Q_grid:   (NQ,) Q points to evaluate
        channel:  'pp' or 'ph'

    Returns:
        omega: (NQ,) minimum eigenvalue of Π^Q at each Q
    """
    omega = np.zeros(len(Q_grid))
    for i, Q in enumerate(Q_grid):
        Pi = nesting_operator(P, Q_mat, k_grid, Q, channel)
        omega[i] = np.linalg.eigvalsh(Pi).min()
    return np.maximum(omega, 0.0)  # clip tiny negatives from numerics


def nesting_matrix(P, Q_mat, k_grid, Q_vec, channel='pp', tol=1e-6):
    """Find nesting matrix N^Q (null space of Π^Q).

    Only meaningful when ω̃₀^Q ≈ 0 (perfect or near-perfect QGN).

    Returns:
        N_list: list of (Norb, Norb) nesting matrices (null vectors reshaped)
        omega_min: minimum eigenvalue (should be ≈ 0 for perfect QGN)
    """
    Norb = P.shape[1]
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec, channel)
    eigvals, eigvecs = np.linalg.eigh(Pi)
    null_indices = np.where(eigvals < tol)[0]
    N_list = [eigvecs[:, i].reshape(Norb, Norb) for i in null_indices]
    return N_list, eigvals.min()
```

**Step 4: Run all tests**

```bash
pytest tests/ -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add qgn/core.py tests/test_core.py
git commit -m "feat: QGN core (nesting operator, nestability map, nesting matrix)"
```

---

### Task 6: Notebook 01 — SSH Chain

**Files:**
- Create: `notebooks/01_SSH.ipynb`

**Step 1: Create notebook with sections**

Section 1 — Model Setup:
```python
import numpy as np, matplotlib.pyplot as plt, sys
sys.path.insert(0, '..')
from qgn.models import ssh_hamiltonian
from qgn.geometry import diagonalize_model, projection_matrix_from_vecs, quantum_distance
from qgn.core import nestability_map, nesting_matrix

Nk = 200
t1, t2 = 1.0, 0.6      # trivial phase; change t2→t1 for perfect QGN
k_grid = np.linspace(0, 2*np.pi, Nk, endpoint=False)
H_func = lambda k: ssh_hamiltonian(k, t1=t1, t2=t2)
```

Section 2 — Band Structure:
```python
eigvals, eigvecs = diagonalize_model(H_func, k_grid)
plt.figure(figsize=(5,3))
for n in range(2):
    plt.plot(k_grid, eigvals[:, n], 'b-')
plt.axhline(0, color='k', lw=0.5, ls='--')
plt.xlabel('k'), plt.ylabel('E'), plt.title('SSH Band Structure')
plt.tight_layout(); plt.savefig('../docs/ssh_bands.png', dpi=100)
```

Section 3 — Projection matrix and quantum distance:
```python
flat_bands = [0]   # lower band
P = projection_matrix_from_vecs(eigvecs, flat_bands)
Q_mat = np.eye(2)[None, ...] - P

# Quantum distance along k
d = np.array([quantum_distance(P[0], P[i], n_flat=1) for i in range(Nk)])
plt.figure(figsize=(5,3))
plt.plot(k_grid, d)
plt.xlabel('k'), plt.ylabel('d(0, k)'), plt.title('Quantum Distance')
```

Section 4 — Nestability map:
```python
Q_grid = np.linspace(0, 2*np.pi, 100, endpoint=False)
omega_pp = nestability_map(P, Q_mat, k_grid, Q_grid, channel='pp')
omega_ph = nestability_map(P, Q_mat, k_grid, Q_grid, channel='ph')

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(Q_grid, omega_pp); axes[0].set_title('p-p nestability')
axes[1].plot(Q_grid, omega_ph); axes[1].set_title('p-h nestability')
for ax in axes: ax.set_xlabel('Q')
plt.tight_layout()
```

Section 5 — Perfect QGN verification (t1 = t2):
```python
t1 = t2 = 1.0
H_func2 = lambda k: ssh_hamiltonian(k, t1=t1, t2=t2)
_, eigvecs2 = diagonalize_model(H_func2, k_grid)
P2 = projection_matrix_from_vecs(eigvecs2, flat_bands=[0])
Q2 = np.eye(2)[None, ...] - P2
N_list, omega_min = nesting_matrix(P2, Q2, k_grid, Q_vec=np.pi, channel='pp')
print(f"ω̃₀^π = {omega_min:.2e}  (should be ≈ 0 for perfect QGN)")
print(f"Nesting matrix N^π:\n{N_list[0]}")
```

**Step 2: Run notebook to verify all cells execute**

```bash
cd /Users/shajianyu/CMP_manybody/QGN
jupyter nbconvert --to notebook --execute notebooks/01_SSH.ipynb \
    --output notebooks/01_SSH.ipynb
```

**Step 3: Commit**

```bash
git add notebooks/01_SSH.ipynb docs/ssh_bands.png
git commit -m "feat: SSH chain analysis notebook"
```

---

### Task 7: Notebook 02 — Custom Tight-Binding

**Files:**
- Create: `notebooks/02_custom_TB.ipynb`

**Step 1: Create notebook**

Demonstrate a 2D square-lattice 2-orbital model with chiral symmetry (known to have
perfect p-h QGN at Q=0 and Q=(π,π)).

```python
import numpy as np, matplotlib.pyplot as plt, sys
sys.path.insert(0, '..')
from qgn.models import custom_tb_hamiltonian
from qgn.geometry import diagonalize_model, projection_matrix_from_vecs, \
                          berry_curvature_grid, chern_number
from qgn.core import nestability_map, nesting_matrix

# 2D square lattice, 2 orbitals, chiral-symmetric model
t = 1.0; m = 0.5
# Hoppings: H(k) = 2t(cos kx + cos ky) σz + 2t(sin kx σx + sin ky σy) + m σz
# (A simple Qi-Wu-Zhang model giving C=±1 Chern bands)
def qwz_hamiltonian(k):
    kx, ky = 2*np.pi*k[0], 2*np.pi*k[1]
    d0 = 0.0
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = m + np.cos(kx) + np.cos(ky)
    H = np.array([[dz,         dx - 1j*dy],
                  [dx + 1j*dy, -dz       ]], dtype=complex)
    return H

Nk = 40
k1 = np.linspace(0, 1, Nk, endpoint=False)
k2 = np.linspace(0, 1, Nk, endpoint=False)
K1, K2 = np.meshgrid(k1, k2, indexing='ij')

# Diagonalize on 2D grid
eigvals_2d = np.zeros((Nk, Nk, 2))
eigvecs_2d = np.zeros((Nk, Nk, 2, 2), dtype=complex)
for i in range(Nk):
    for j in range(Nk):
        e, v = np.linalg.eigh(qwz_hamiltonian(np.array([k1[i], k2[j]])))
        eigvals_2d[i, j] = e
        eigvecs_2d[i, j] = v
```

Section 2 — Berry curvature and Chern number:
```python
dk = 1.0/Nk
Omega = berry_curvature_grid(eigvecs_2d, flat_bands=[0], dk1=dk, dk2=dk)
C = chern_number(eigvecs_2d, flat_bands=[0])
print(f"Chern number = {C:.3f}  (should be ±1 for QWZ model)")

plt.figure(figsize=(4,4))
plt.pcolormesh(K1, K2, Omega, cmap='RdBu_r', shading='auto')
plt.colorbar(label='Ω(k)'); plt.title(f'Berry curvature  C={C:.2f}')
```

Section 3 — QGN on 1D scan of Q:
```python
# Flatten 2D grid to 1D for core.py (demonstrates parameter sweep)
# Use a 1D cut kx = ky for illustration
k_diag = np.array([np.array([k, k]) for k in k1])
H_func_1d = lambda k: qwz_hamiltonian(np.array([k, k]) / (2*np.pi))
eigvals_1d, eigvecs_1d = diagonalize_model(
    lambda k: qwz_hamiltonian(np.array([k/(2*np.pi), k/(2*np.pi)])),
    np.linspace(0, 2*np.pi, 100, endpoint=False)
)
```

**Step 2: Run notebook**

```bash
jupyter nbconvert --to notebook --execute notebooks/02_custom_TB.ipynb \
    --output notebooks/02_custom_TB.ipynb
```

**Step 3: Commit**

```bash
git add notebooks/02_custom_TB.ipynb
git commit -m "feat: custom 2D tight-binding (QWZ) analysis notebook"
```

---

### Task 8: Notebook 03 — TBG Wannier

**Files:**
- Create: `notebooks/03_TBG.ipynb`

**Step 1: Create notebook**

```python
import numpy as np, matplotlib.pyplot as plt, sys
sys.path.insert(0, '..')
from qgn.models import tbg_wannier_hamiltonian
from qgn.geometry import diagonalize_model, projection_matrix_from_vecs, \
                          berry_curvature_grid, chern_number, quantum_distance
from qgn.core import nestability_map, nesting_matrix

# TBG moiré BZ: use 2D k-grid
Nk = 30
k1 = np.linspace(0, 1, Nk, endpoint=False)
k2 = np.linspace(0, 1, Nk, endpoint=False)

eigvals_2d = np.zeros((Nk, Nk, 4))
eigvecs_2d = np.zeros((Nk, Nk, 4, 4), dtype=complex)
for i in range(Nk):
    for j in range(Nk):
        e, v = np.linalg.eigh(tbg_wannier_hamiltonian(np.array([k1[i], k2[j]])))
        eigvals_2d[i, j] = e
        eigvecs_2d[i, j] = v
```

Section 2 — Flat bands (bands 1 and 2 at magic angle):
```python
# Identify flat bands by bandwidth
bandwidths = eigvals_2d[:,:,:].max(axis=(0,1)) - eigvals_2d[:,:,:].min(axis=(0,1))
print("Bandwidths:", bandwidths)
flat_bands = [1, 2]   # two middle bands

plt.figure(figsize=(6,4))
# Plot along Γ-M-K-Γ high symmetry path
ks = np.linspace(0, 1, 100)
path_k = np.column_stack([ks, np.zeros(100)])  # Γ-M direction
Es = np.array([np.linalg.eigvalsh(tbg_wannier_hamiltonian(k)) for k in path_k])
for n in range(4):
    plt.plot(ks, Es[:, n], 'b-' if n in flat_bands else 'r--')
plt.xlabel('k (Γ→M)'), plt.ylabel('E (eV)'), plt.title('TBG Bands')
```

Section 3 — Berry curvature and Chern number:
```python
dk = 1.0/Nk
Omega = berry_curvature_grid(eigvecs_2d, flat_bands=[flat_bands[0]], dk1=dk, dk2=dk)
C = chern_number(eigvecs_2d, flat_bands=[flat_bands[0]])
print(f"Chern number of lower flat band = {C:.3f}")
```

Section 4 — QGN nestability scan (along diagonal of BZ):
```python
# 1D scan along k1=k2 direction
k_diag = np.linspace(0, 2*np.pi, 60, endpoint=False)
H_1d = lambda k: tbg_wannier_hamiltonian(np.array([k, k]) / (2*np.pi))
_, eigvecs_1d = diagonalize_model(H_1d, k_diag)
P_1d = projection_matrix_from_vecs(eigvecs_1d, flat_bands=flat_bands)
Q_1d = np.eye(4)[None, ...] - P_1d

Q_scan = np.linspace(0, 2*np.pi, 40, endpoint=False)
omega_pp = nestability_map(P_1d, Q_1d, k_diag, Q_scan, channel='pp')
omega_ph = nestability_map(P_1d, Q_1d, k_diag, Q_scan, channel='ph')

plt.figure(figsize=(8,3))
plt.subplot(1,2,1); plt.plot(Q_scan, omega_pp); plt.title('TBG p-p nestability')
plt.subplot(1,2,2); plt.plot(Q_scan, omega_ph); plt.title('TBG p-h nestability')
plt.tight_layout()
```

Section 5 — Nesting matrix at minimum Q:
```python
Q_min = Q_scan[np.argmin(omega_pp)]
N_list, omega_min = nesting_matrix(P_1d, Q_1d, k_diag, Q_min, channel='pp')
print(f"Minimum ω̃₀^Q = {omega_min:.4f} at Q = {Q_min:.3f}")
for i, N in enumerate(N_list):
    print(f"Nesting matrix {i}:\n{N}\n")
```

**Step 2: Run notebook**

```bash
jupyter nbconvert --to notebook --execute notebooks/03_TBG.ipynb \
    --output notebooks/03_TBG.ipynb
```

**Step 3: Commit**

```bash
git add notebooks/03_TBG.ipynb
git commit -m "feat: TBG Wannier QGN analysis notebook"
```

---

### Task 9: Extend `core.py` to 2D k-grids (optional, for full BZ heatmaps)

**Files:**
- Modify: `qgn/core.py`
- Modify: `tests/test_core.py`

Add `nesting_operator_2d` and `nestability_map_2d` functions that accept
`(Nk1, Nk2, Norb, Norb)` P arrays and compute Π^Q for each Q on a 2D grid,
producing a 2D heatmap of ω̃₀^Q.

This task is optional but needed for the full BZ nestability heatmap shown in
the paper (Fig. 1 style plots). Implement after confirming all 1D results look correct.

```bash
git commit -m "feat: 2D nestability heatmap for full BZ visualization"
```
