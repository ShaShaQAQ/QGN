import numpy as np
from qgn.models import ssh_hamiltonian, custom_tb_hamiltonian, tbg_wannier_hamiltonian


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
