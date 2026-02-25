import numpy as np
from qgn.models import ssh_hamiltonian
from qgn.geometry import (diagonalize_model, projection_matrix_from_vecs,
                           berry_curvature_grid, chern_number, quantum_distance)


def _ssh_data(Nk=50, t1=1.0, t2=0.5):
    k_grid = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1, t2)
    eigvals, eigvecs = diagonalize_model(H_func, k_grid)
    return k_grid, eigvals, eigvecs


# --- diagonalize_model ---

def test_diagonalize_returns_correct_shapes():
    k_grid = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1=1.0, t2=0.5)
    eigvals, eigvecs = diagonalize_model(H_func, k_grid)
    assert eigvals.shape == (30, 2)
    assert eigvecs.shape == (30, 2, 2)


def test_diagonalize_eigenvalues_sorted():
    k_grid, eigvals, _ = _ssh_data(Nk=20)
    assert np.all(eigvals[:, 0] <= eigvals[:, 1])


# --- projection_matrix_from_vecs ---

def test_projection_idempotent():
    _, _, eigvecs = _ssh_data(Nk=20)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    # P^2 = P
    PP = np.einsum('kij,kjl->kil', P, P)
    np.testing.assert_allclose(PP, P, atol=1e-12)


def test_projection_hermitian():
    _, _, eigvecs = _ssh_data(Nk=20)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    np.testing.assert_allclose(P, P.conj().transpose(0, 2, 1), atol=1e-12)


def test_projection_trace():
    # tr[P(k)] = number of flat bands
    _, _, eigvecs = _ssh_data(Nk=20)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    traces = np.trace(P, axis1=1, axis2=2).real
    np.testing.assert_allclose(traces, 1.0, atol=1e-12)


# --- quantum_distance ---

def test_quantum_distance_self_zero():
    _, _, eigvecs = _ssh_data(Nk=20)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    d = quantum_distance(P[0], P[0], n_flat=1)
    # sqrt amplifies floating-point errors; numerical tolerance ~1e-7 is expected
    np.testing.assert_allclose(d, 0.0, atol=1e-6)


def test_quantum_distance_bounded():
    _, _, eigvecs = _ssh_data(Nk=20)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    d = quantum_distance(P[0], P[5], n_flat=1)
    assert 0.0 <= d <= 1.0


# --- berry_curvature_grid and chern_number (2D QWZ model) ---

def _qwz_eigvecs_2d(Nk=20, m=0.5):
    """QWZ model eigenvectors on 2D k-grid."""
    def qwz(k):
        kx, ky = 2 * np.pi * k[0], 2 * np.pi * k[1]
        dx = np.sin(kx)
        dy = np.sin(ky)
        dz = m + np.cos(kx) + np.cos(ky)
        return np.array([[dz, dx - 1j * dy],
                         [dx + 1j * dy, -dz]], dtype=complex)

    k1 = np.linspace(0, 1, Nk, endpoint=False)
    k2 = np.linspace(0, 1, Nk, endpoint=False)
    eigvecs_2d = np.zeros((Nk, Nk, 2, 2), dtype=complex)
    for i in range(Nk):
        for j in range(Nk):
            _, v = np.linalg.eigh(qwz(np.array([k1[i], k2[j]])))
            eigvecs_2d[i, j] = v
    return eigvecs_2d


def test_berry_curvature_shape():
    eigvecs_2d = _qwz_eigvecs_2d(Nk=20)
    dk = 1.0 / 20
    Omega = berry_curvature_grid(eigvecs_2d, flat_bands=[0], dk1=dk, dk2=dk)
    assert Omega.shape == (20, 20)


def test_chern_number_qwz():
    # QWZ model with m=0.5 (0 < m < 2): |Chern number| = 1
    # The lower band (flat_bands=[0]) has C = +1 in the standard numpy eigh convention
    eigvecs_2d = _qwz_eigvecs_2d(Nk=30, m=0.5)
    C = chern_number(eigvecs_2d, flat_bands=[0])
    np.testing.assert_allclose(abs(C), 1.0, atol=0.05)
