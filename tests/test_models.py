import numpy as np
from qgn.models import (ssh_hamiltonian, custom_tb_hamiltonian,
                        tbg_wannier_hamiltonian, fci_triangular_hamiltonian)


def test_ssh_shape():
    H = ssh_hamiltonian(k=0.0, t1=1.0, t2=0.5)
    assert H.shape == (2, 2)


def test_ssh_hermitian():
    for k in np.linspace(0, 2 * np.pi, 20):
        H = ssh_hamiltonian(k=k, t1=1.0, t2=0.5)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_ssh_spectrum_symmetric():
    # SSH spectrum is symmetric: ε_n(k) = -ε_n(k) (chiral symmetry)
    k = 0.7
    eigvals = np.linalg.eigvalsh(ssh_hamiltonian(k, t1=1.0, t2=0.5))
    np.testing.assert_allclose(eigvals[0], -eigvals[1], atol=1e-12)


# --- custom_tb_hamiltonian ---

def test_custom_tb_shape():
    # 1-orbital model: hopping to NN at R=(1,0) and R=(-1,0)
    hoppings = {
        (1, 0): np.array([[0.5]]),
        (-1, 0): np.array([[0.5]]),   # Hermitian conjugate
        (0, 1): np.array([[0.3]]),
        (0, -1): np.array([[0.3]]),
    }
    H = custom_tb_hamiltonian(k=np.array([0.1, 0.2]), hoppings=hoppings, norb=1)
    assert H.shape == (1, 1)


def test_custom_tb_hermitian():
    # For a properly specified hopping dict (R and -R both present), H must be Hermitian
    t = np.array([[0.5 + 0.1j]])
    hoppings = {(1, 0): t, (-1, 0): t.conj().T}
    for kx in np.linspace(0, 1, 10):
        H = custom_tb_hamiltonian(np.array([kx, 0.0]), hoppings, norb=1)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


# --- tbg_wannier_hamiltonian ---

def test_tbg_shape():
    H = tbg_wannier_hamiltonian(k=np.array([0.0, 0.0]))
    assert H.shape == (4, 4)


def test_tbg_hermitian():
    rng = np.random.default_rng(42)
    for _ in range(5):
        k = rng.random(2)
        H = tbg_wannier_hamiltonian(k)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


# --- fci_triangular_hamiltonian  (Kourtis et al. PRB 86, 235118, Eq. 23) ---

def test_fci_shape():
    H = fci_triangular_hamiltonian(k=np.array([0.0, 0.0]))
    assert H.shape == (2, 2)


def test_fci_hermitian():
    rng = np.random.default_rng(0)
    for _ in range(10):
        k = rng.uniform(-np.pi, np.pi, size=2)
        H = fci_triangular_hamiltonian(k, t=1.0, tp=0.2)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_fci_dispersion():
    """Eigenvalues must match the analytical dispersion Eq. (24):
    ε± = ±2t √(Σ_j cos²(k·a_j)) + 2t' Σ_j cos(2k·a_j)
    """
    t, tp = 1.0, 0.2
    a1 = np.array([1.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2])
    a3 = np.array([-0.5, -np.sqrt(3) / 2])

    rng = np.random.default_rng(7)
    for _ in range(10):
        k = rng.uniform(-np.pi, np.pi, size=2)
        H = fci_triangular_hamiltonian(k, t=t, tp=tp)
        eigvals = np.linalg.eigvalsh(H)

        f = np.array([np.cos(k @ a1), np.cos(k @ a2), np.cos(k @ a3)])
        d0 = 2 * tp * np.sum(np.cos(2 * np.array([k @ a1, k @ a2, k @ a3])))
        eps_pm = np.sort([d0 - 2 * t * np.sqrt(np.sum(f**2)),
                          d0 + 2 * t * np.sqrt(np.sum(f**2))])
        np.testing.assert_allclose(eigvals, eps_pm, atol=1e-12)


def test_fci_chern_number():
    """Lower band of FCI model must have |Chern number| = 1.

    The 2-site unit cell has primitive lattice vectors
    A1 = (3/2, √3/2), A2 = (0, √3), so the correct reciprocal vectors are
    B1 = (4π/3, 0), B2 = (-2π/3, 2π/√3).
    """
    from qgn.geometry import chern_number

    t, tp = 1.0, 0.2
    # Reciprocal vectors of the 2-site magnetic unit cell
    b1 = np.array([4 * np.pi / 3, 0.0])
    b2 = np.array([-2 * np.pi / 3, 2 * np.pi / np.sqrt(3)])

    Nk = 25
    k1 = np.linspace(0, 1, Nk, endpoint=False)
    k2 = np.linspace(0, 1, Nk, endpoint=False)
    eigvecs_2d = np.zeros((Nk, Nk, 2, 2), dtype=complex)
    for i in range(Nk):
        for j in range(Nk):
            k_phys = k1[i] * b1 + k2[j] * b2
            _, v = np.linalg.eigh(fci_triangular_hamiltonian(k_phys, t=t, tp=tp))
            eigvecs_2d[i, j] = v

    C = chern_number(eigvecs_2d, flat_bands=[0])
    np.testing.assert_allclose(abs(C), 1.0, atol=0.05)
