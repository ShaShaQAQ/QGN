import numpy as np


def ssh_hamiltonian(k, t1=1.0, t2=0.5):
    """SSH chain Hamiltonian (1D, 2 orbitals per unit cell).

    H(k) = [[0,               t1 + t2*exp(-ik)],
            [t1 + t2*exp(ik), 0               ]]

    Perfect QGN at Q=π when |t1| == |t2|.
    """
    h01 = t1 + t2 * np.exp(-1j * k)
    return np.array([[0, h01], [h01.conj(), 0]], dtype=complex)


def custom_tb_hamiltonian(k, hoppings, norb):
    """Generic tight-binding Hamiltonian from hopping dictionary.

    Args:
        k: wave vector in fractional coordinates, shape (2,) for 2D
        hoppings: dict {R: t_matrix} where R is lattice vector tuple,
                  t_matrix is (norb, norb) complex array.
                  Convention: H(k) = Σ_R t(R) exp(i 2π k·R).
                  Must include both R and -R for off-diagonal terms.
        norb: number of orbitals per unit cell

    Returns:
        H: (norb, norb) Hamiltonian matrix
    """
    H = np.zeros((norb, norb), dtype=complex)
    for R, t in hoppings.items():
        phase = np.exp(1j * 2 * np.pi * np.dot(k, np.array(R)))
        H += np.array(t, dtype=complex) * phase
    return H


# TBG moiré lattice basis vectors (in moiré units)
_TBG_A1 = np.array([1.0, 0.0])
_TBG_A2 = np.array([0.5, np.sqrt(3) / 2])

# Nearest-neighbor vectors of the moiré triangular lattice
_TBG_NN = [_TBG_A1, _TBG_A2, _TBG_A2 - _TBG_A1]

# Simplified Wannier parameters (eV), 4-orbital basis: (A1, B1, A2, B2)
# where 1/2 label moiré sublattice and A/B label layer-like index
_TBG_PARAMS = {
    't1': -0.331,  # nearest-neighbor same-sublattice hopping
    't2':  0.368,  # nearest-neighbor cross-sublattice hopping
    't3': -0.026,  # next-nearest-neighbor hopping (not used in minimal model)
}


def tbg_wannier_hamiltonian(k, params=None):
    """TBG Wannier tight-binding Hamiltonian (4-band, valley K).

    4 orbitals: (A1, B1, A2, B2) — two sublattice-like Wannier centers
    on the moiré triangular lattice.

    Args:
        k: wave vector in moiré BZ fractional coordinates, shape (2,)
        params: dict with keys 't1', 't2', 't3'. Defaults to _TBG_PARAMS.

    Returns:
        H: (4, 4) Hermitian matrix
    """
    p = params if params is not None else _TBG_PARAMS
    t1, t2 = p['t1'], p['t2']

    # Structure factor: sum of phase factors for nearest-neighbor vectors
    f = sum(np.exp(1j * 2 * np.pi * np.dot(k, nn)) for nn in _TBG_NN)
    f_conj = np.conj(f)

    # 2×2 blocks in basis (A1, B1) or (A2, B2)
    h_intra = np.array([[0,       t2 * f],
                        [t2 * f_conj, 0  ]], dtype=complex)
    h_inter = t1 * np.eye(2, dtype=complex)

    H = np.block([[h_intra,           h_inter],
                  [h_inter.conj().T,  h_intra]])
    return H


# ── FCI triangular lattice model ──────────────────────────────────────────────
# Three NN bond vectors of the triangular lattice (lattice constant = 1)
# They satisfy a1 + a2 + a3 = 0.
_FCI_A1 = np.array([1.0, 0.0])
_FCI_A2 = np.array([-0.5,  np.sqrt(3) / 2])
_FCI_A3 = np.array([-0.5, -np.sqrt(3) / 2])

_SX = np.array([[0, 1],   [1,  0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0],   [0, -1]], dtype=complex)


def fci_triangular_hamiltonian(k, t=1.0, tp=0.2):
    """Two-band tight-binding model for FCI on the triangular lattice.

    Eq. (23) of Kourtis, Venderbos & Daghofer, PRB 86, 235118 (2012):

        H_kin(k) = 2t Σ_j σ^j cos(k·a_j)  +  2t' Σ_j σ^0 cos(2k·a_j)

    where σ^{1,2,3} = σ_{x,y,z} act in the 2-site unit-cell space, and
    a_{1,2,3} are the three NN bond vectors of the triangular lattice
    (a_1 + a_2 + a_3 = 0).

    Dispersion (Eq. 24):
        ε^± = ±2t √(Σ_j cos²(k·a_j))  +  2t' Σ_j cos(2k·a_j)

    Lower band has Chern number C = ±1.
    Near-flat bands are achieved for t'/t ≈ 0.2.

    Args:
        k:  Cartesian wave vector, shape (2,)
        t:  NN hopping (energy unit)
        tp: third-neighbor hopping t'

    Returns:
        H: (2, 2) Hermitian matrix
    """
    f1 = np.cos(k @ _FCI_A1)
    f2 = np.cos(k @ _FCI_A2)
    f3 = np.cos(k @ _FCI_A3)
    g  = (np.cos(2 * k @ _FCI_A1) +
          np.cos(2 * k @ _FCI_A2) +
          np.cos(2 * k @ _FCI_A3))

    H = 2 * t * (f1 * _SX + f2 * _SY + f3 * _SZ) + 2 * tp * g * np.eye(2, dtype=complex)
    return H


# ── Haldane honeycomb model ────────────────────────────────────────────────────
# Lattice: honeycomb, bond length = 1.
# Primitive vectors (real-space):
#   a1 = (√3, 0),  a2 = (√3/2, 3/2)
# Reciprocal lattice vectors (satisfy ai·bj = 2π δij):
#   b1 = (2π/√3, -2π/3),  b2 = (0, 4π/3)
# NN vectors from A→B sublattice:
#   δ1 = (0, 1),  δ2 = (√3/2, -1/2),  δ3 = (−√3/2, −1/2)
# NNN vectors (CCW for A sublattice):
#   v1 = a1,  v2 = a2,  v3 = a2 − a1
_HAL_A1 = np.array([np.sqrt(3), 0.0])
_HAL_A2 = np.array([np.sqrt(3) / 2, 1.5])
_HAL_D1 = np.array([0.0, 1.0])
_HAL_D2 = np.array([np.sqrt(3) / 2, -0.5])
_HAL_D3 = np.array([-np.sqrt(3) / 2, -0.5])
_HAL_V1 = _HAL_A1
_HAL_V2 = _HAL_A2
_HAL_V3 = _HAL_A2 - _HAL_A1

# Reciprocal lattice vectors (for BZ sampling in notebook)
HAL_B1 = np.array([2 * np.pi / np.sqrt(3), -2 * np.pi / 3])
HAL_B2 = np.array([0.0, 4 * np.pi / 3])


def haldane_hamiltonian(k, t1=1.0, t2=0.3, phi=np.pi / 2, M=0.0):
    """Haldane honeycomb model (Haldane 1988).

    H(k) = [[H_AA, h_AB],
             [h_AB*, H_BB]]

    where:
        h_AB = t1 * Σ_j exp(i k·δ_j)             (NN, off-diagonal)
        H_AA = +M + 2t2 Σ_i cos(k·v_i + φ)       (NNN for A sublattice)
        H_BB = −M + 2t2 Σ_i cos(k·v_i − φ)       (NNN for B sublattice, opposite chirality)

    Lattice: honeycomb, bond length 1.
    Primitive vectors: a1=(√3,0), a2=(√3/2,3/2).
    NNN vectors (CCW from A): v1=a1, v2=a2, v3=a2−a1.

    Topological phase (C = +1) for |M| < 3√3 t2 |sin φ|.
    With M=0, φ=π/2: always in C=+1 phase for t2≠0.

    Args:
        k:   Cartesian wave vector, shape (2,)
        t1:  NN hopping (energy unit)
        t2:  NNN hopping
        phi: NNN phase (π/2 gives maximal gap and C=+1)
        M:   sublattice staggered potential (breaks inversion)

    Returns:
        H: (2, 2) Hermitian matrix
    """
    h_AB = t1 * (np.exp(1j * np.dot(k, _HAL_D1))
                 + np.exp(1j * np.dot(k, _HAL_D2))
                 + np.exp(1j * np.dot(k, _HAL_D3)))

    nnn_A = 2 * t2 * (np.cos(np.dot(k, _HAL_V1) + phi)
                      + np.cos(np.dot(k, _HAL_V2) + phi)
                      + np.cos(np.dot(k, _HAL_V3) + phi))
    nnn_B = 2 * t2 * (np.cos(np.dot(k, _HAL_V1) - phi)
                      + np.cos(np.dot(k, _HAL_V2) - phi)
                      + np.cos(np.dot(k, _HAL_V3) - phi))

    H_AA = M + nnn_A
    H_BB = -M + nnn_B
    return np.array([[H_AA, h_AB], [np.conj(h_AB), H_BB]], dtype=complex)


# ── QWZ square-lattice Chern insulator ────────────────────────────────────────
# Qi, Wu, Zhang (2006) — simplest lattice model of a 2D Chern insulator.
# Physically related to the Tešanović (1989) flux-phase insulator (review §2.3):
# both are 2-band square-lattice models with C=±1, no external magnetic field.
#
# BZ: kx, ky ∈ (−π, π].  Primitive vectors a1=(1,0), a2=(0,1).
# Reciprocal lattice: B1=(2π,0), B2=(0,2π).
QWZ_B1 = np.array([2 * np.pi, 0.0])
QWZ_B2 = np.array([0.0, 2 * np.pi])


def qwz_hamiltonian(k, m=1.0):
    """Qi–Wu–Zhang (2006) Chern insulator on a square lattice.

    H(k) = sin(kx)·σx + sin(ky)·σy + (2 − m − cos(kx) − cos(ky))·σz

    Chern number of the lower band:
        C = +1  for  0 < m < 2
        C = −1  for  2 < m < 4
        C =  0  otherwise

    Related to the Tešanović (1989) flux-phase model (review §2.3): both are
    2-band square-lattice models with C=±1 bands and spontaneous AHE (σ_xy=e²/h).

    Args:
        k: Cartesian wave vector (kx, ky), BZ ∈ (−π, π]², shape (2,)
        m: mass parameter (default 1.0 → C=+1 phase)

    Returns:
        H: (2, 2) Hermitian matrix
    """
    kx, ky = k[0], k[1]
    dz  = 2.0 - m - np.cos(kx) - np.cos(ky)
    hAB = np.sin(kx) + 1j * np.sin(ky)
    return np.array([[dz, hAB], [np.conj(hAB), -dz]], dtype=complex)
