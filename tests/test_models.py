import numpy as np
from qgn.models import ssh_hamiltonian


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
