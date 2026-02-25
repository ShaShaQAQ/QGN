import numpy as np


def diagonalize_model(H_func, k_points):
    """Diagonalize H(k) at each k point.

    Args:
        H_func: callable k -> H[Norb, Norb]; k can be scalar (1D) or array (2D)
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

    P_μν(k) = Σ_{n ∈ flat_bands} U_μn(k) U†_nν(k)   [Eq. 4 in PhysRevX.14.041004]

    Args:
        eigvecs: (Nk, Norb, Norb) — eigvecs[k, :, n] is the n-th Bloch vector
        flat_bands: list of band indices forming the flat-band subspace

    Returns:
        P: (Nk, Norb, Norb) projection matrices
    """
    U_flat = eigvecs[:, :, flat_bands]                       # (Nk, Norb, N_flat)
    P = U_flat @ U_flat.conj().transpose(0, 2, 1)            # (Nk, Norb, Norb)
    return P


def berry_curvature_grid(eigvecs, flat_bands, dk1, dk2):
    """Berry curvature on a 2D k-grid using the projection matrix formula.

    Ω(k) = -2 Im tr[ P ∂_{k1} P ∂_{k2} P ]

    Args:
        eigvecs: (Nk1, Nk2, Norb, Norb)
        flat_bands: list of band indices
        dk1, dk2: grid spacing in k1 and k2 directions

    Returns:
        Omega: (Nk1, Nk2) Berry curvature array
    """
    Nk1, Nk2 = eigvecs.shape[:2]
    P = projection_matrix_from_vecs(
        eigvecs.reshape(Nk1 * Nk2, *eigvecs.shape[2:]), flat_bands
    ).reshape(Nk1, Nk2, *eigvecs.shape[2:])                  # (Nk1, Nk2, Norb, Norb)

    dP1 = np.gradient(P, dk1, axis=0)
    dP2 = np.gradient(P, dk2, axis=1)

    # tr[P ∂1P ∂2P] at each k point
    product = np.einsum('...ij,...jk,...ki->...', P, dP1, dP2)
    return -2.0 * np.imag(product)


def chern_number(eigvecs_2d, flat_bands):
    """Chern number via Fukui-Hatsugai-Suzuki lattice method.

    Args:
        eigvecs_2d: (Nk1, Nk2, Norb, Norb) eigenvectors on 2D grid
        flat_bands: list of band indices

    Returns:
        C: float, close to an integer for a Chern insulator
    """
    Nk1, Nk2 = eigvecs_2d.shape[:2]
    F_total = 0.0
    for i in range(Nk1):
        for j in range(Nk2):
            U00 = eigvecs_2d[i,            j,            :, :][:, flat_bands]
            U10 = eigvecs_2d[(i + 1) % Nk1, j,           :, :][:, flat_bands]
            U11 = eigvecs_2d[(i + 1) % Nk1, (j + 1) % Nk2, :, :][:, flat_bands]
            U01 = eigvecs_2d[i,            (j + 1) % Nk2, :, :][:, flat_bands]
            # Link variables (U† U)
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
