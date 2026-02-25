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
