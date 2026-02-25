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
    # Extend P periodically: append one period on each side for interpolation
    k_ext = np.concatenate([k_grid - 2 * np.pi, k_grid, k_grid + 2 * np.pi])
    P_ext = np.concatenate([P, P, P], axis=0)

    P_shifted = np.zeros_like(P)
    for mu in range(Norb):
        for nu in range(Norb):
            f_re = interp1d(k_ext, P_ext[:, mu, nu].real, kind='linear')
            f_im = interp1d(k_ext, P_ext[:, mu, nu].imag, kind='linear')
            P_shifted[:, mu, nu] = f_re(k_mod) + 1j * f_im(k_mod)
    return P_shifted


def nesting_operator(P, Q_mat, k_grid, Q_vec, channel='pp'):
    """Compute nesting operator Π^Q.  [Eq. 5/6 in PhysRevX.14.041004]

    Supports 1D k_grid (scalar Q_vec). For 2D see nesting_operator_2d.

    Convention for the pp channel:
        Π^Q_{μ'ν';μν} = (1/Nk) Σ_k [ P*_{μ'μ}(k+Q/2) Q_{νν'}(k-Q/2)
                                      + Q*_{μ'μ}(k+Q/2) P_{νν'}(k-Q/2) ]

    Compound index ordering: row = (μ', ν'), col = (μ, ν).

    Args:
        P:      (Nk, Norb, Norb) flat-band projection matrices
        Q_mat:  (Nk, Norb, Norb) complement projector Q = I - P
        k_grid: (Nk,) k points
        Q_vec:  scalar nesting wave vector
        channel: 'pp' (particle-particle) or 'ph' (particle-hole)

    Returns:
        Pi: (Norb², Norb²) Hermitian positive-semidefinite matrix
    """
    Nk, Norb, _ = P.shape

    k_plus = k_grid + Q_vec / 2
    k_minus = k_grid - Q_vec / 2

    P_plus  = _interpolate_P_1d(P,     k_grid, k_plus)   # P(k + Q/2)
    P_minus = _interpolate_P_1d(P,     k_grid, k_minus)  # P(k - Q/2)
    Q_plus  = _interpolate_P_1d(Q_mat, k_grid, k_plus)   # Q(k + Q/2)
    Q_minus = _interpolate_P_1d(Q_mat, k_grid, k_minus)  # Q(k - Q/2)

    if channel == 'pp':
        # term1[k, μ', ν, μ, ν'] = P*_{μ'μ}(k+Q/2) * Q_{νν'}(k-Q/2)
        # einsum: P_plus.conj()[k,a,b] * Q_minus[k,c,d] -> [k,a,c,b,d]
        term1 = np.einsum('kab,kcd->kacbd', P_plus.conj(), Q_minus)
        # term2[k, μ', ν, μ, ν'] = Q*_{μ'μ}(k+Q/2) * P_{νν'}(k-Q/2)
        term2 = np.einsum('kab,kcd->kacbd', Q_plus.conj(), P_minus)
    else:  # ph channel
        # term1[k, μ', ν, μ, ν'] = P_{μ'μ}(k+Q/2) * Q_{νν'}(k-Q/2)
        term1 = np.einsum('kab,kcd->kacbd', P_plus, Q_minus)
        # term2[k, μ', ν, μ, ν'] = Q_{μ'μ}(k+Q/2) * P_{νν'}(k-Q/2)
        term2 = np.einsum('kab,kcd->kacbd', Q_plus, P_minus)

    # Sum over k and reshape to (Norb², Norb²)
    # Pi_raw[μ', ν, μ, ν'] summed over k -> reshape to [(μ'ν), (μν')]
    Pi_raw = (term1 + term2).sum(axis=0) / Nk   # (Norb, Norb, Norb, Norb)
    Pi = Pi_raw.reshape(Norb ** 2, Norb ** 2)
    # Symmetrize to suppress numerical asymmetry
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
        omega: (NQ,) minimum eigenvalue of Π^Q at each Q (clipped to ≥ 0)
    """
    omega = np.zeros(len(Q_grid))
    for i, Q in enumerate(Q_grid):
        Pi = nesting_operator(P, Q_mat, k_grid, Q, channel)
        omega[i] = np.linalg.eigvalsh(Pi).min()
    return np.maximum(omega, 0.0)   # clip tiny numerical negatives


def nesting_matrix(P, Q_mat, k_grid, Q_vec, channel='pp', tol=1e-4):
    """Find nesting matrix N^Q from the null space of Π^Q.

    Only meaningful when ω̃₀^Q ≈ 0 (perfect or near-perfect QGN).

    Args:
        P, Q_mat: (Nk, Norb, Norb)
        k_grid:   (Nk,) 1D k points
        Q_vec:    scalar nesting wave vector
        channel:  'pp' or 'ph'
        tol:      eigenvalue threshold for null-space detection

    Returns:
        N_list:   list of (Norb, Norb) nesting matrices (null vectors reshaped)
        omega_min: minimum eigenvalue (≈ 0 for perfect QGN)
    """
    Norb = P.shape[1]
    Pi = nesting_operator(P, Q_mat, k_grid, Q_vec, channel)
    eigvals, eigvecs = np.linalg.eigh(Pi)
    null_indices = np.where(eigvals < tol)[0]
    N_list = [eigvecs[:, i].reshape(Norb, Norb) for i in null_indices]
    return N_list, eigvals.min()
