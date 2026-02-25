import numpy as np
from qgn.models import ssh_hamiltonian
from qgn.geometry import diagonalize_model, projection_matrix_from_vecs
from qgn.core import nesting_operator, nestability_map, nesting_matrix


def _get_ssh_P(Nk=60, t1=1.0, t2=0.5):
    k_grid = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    H_func = lambda k: ssh_hamiltonian(k, t1, t2)
    _, eigvecs = diagonalize_model(H_func, k_grid)
    P = projection_matrix_from_vecs(eigvecs, flat_bands=[0])
    Q_mat = np.eye(2)[None, ...] - P
    return k_grid, P, Q_mat


# --- nesting_operator ---

def test_nesting_operator_shape():
    k_grid, P, Q_mat = _get_ssh_P()
    Norb = 2
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp')
    assert Pi.shape == (Norb ** 2, Norb ** 2)


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


def test_nesting_operator_ph_hermitian():
    k_grid, P, Q_mat = _get_ssh_P()
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='ph')
    np.testing.assert_allclose(Pi, Pi.conj().T, atol=1e-10)


def test_nesting_operator_ph_psd():
    k_grid, P, Q_mat = _get_ssh_P()
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=np.pi, channel='ph')
    eigvals = np.linalg.eigvalsh(Pi)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


# --- perfect QGN: flat-band atomic limit ---

def _flat_band_P(Nk=60):
    """P(k) = diag(1, 0) for all k (trivially flat band, perfect QGN at any Q)."""
    P = np.zeros((Nk, 2, 2), dtype=complex)
    P[:, 0, 0] = 1.0
    Q_mat = np.zeros((Nk, 2, 2), dtype=complex)
    Q_mat[:, 1, 1] = 1.0
    k_grid = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
    return k_grid, P, Q_mat


def test_atomic_flat_band_perfect_qgn():
    """Atomic flat band (P(k) = const) must have ω̃₀^Q = 0 at any Q."""
    k_grid, P, Q_mat = _flat_band_P(Nk=60)
    for Q in [0.0, np.pi / 2, np.pi]:
        Pi = nesting_operator(P, Q_mat, k_grid, Q_vec=Q, channel='pp')
        omega = np.linalg.eigvalsh(Pi).min()
        np.testing.assert_allclose(omega, 0.0, atol=1e-12,
                                   err_msg=f"Flat band should have ω̃₀^Q=0 at Q={Q:.3f}")


# --- nestability_map ---

def test_nestability_map_shape():
    k_grid, P, Q_mat = _get_ssh_P(Nk=40)
    Q_grid = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    omega_map = nestability_map(P, Q_mat, k_grid, Q_grid, channel='pp')
    assert omega_map.shape == (20,)


def test_nestability_map_non_negative():
    k_grid, P, Q_mat = _get_ssh_P(Nk=40)
    Q_grid = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    omega_map = nestability_map(P, Q_mat, k_grid, Q_grid, channel='pp')
    assert np.all(omega_map >= -1e-10)


# --- nesting_matrix ---

def test_nesting_matrix_flat_band():
    """For atomic flat band, nesting_matrix must return non-empty N_list and omega_min≈0."""
    k_grid, P, Q_mat = _flat_band_P(Nk=60)
    N_list, omega_min = nesting_matrix(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp', tol=1e-8)
    assert len(N_list) >= 1
    np.testing.assert_allclose(omega_min, 0.0, atol=1e-12)


def test_nesting_matrix_ssh_returns_list():
    """nesting_matrix should run without error on SSH model (may return empty list)."""
    k_grid, P, Q_mat = _get_ssh_P(Nk=60)
    N_list, omega_min = nesting_matrix(P, Q_mat, k_grid, Q_vec=np.pi, channel='pp', tol=0.5)
    assert isinstance(N_list, list)
    assert omega_min >= 0.0
