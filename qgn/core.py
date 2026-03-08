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
        # Π^{pp,Q}_{μ'ν';μν} = P*_{μ'μ}(k+Q/2) Q_{νν'}(k-Q/2) + (P↔Q)
        # row compound = (μ',ν') = (a,d), col compound = (μ,ν) = (b,c)
        # einsum: P*.conj()[k,a,b] * Q[k,c,d] -> [k, a, d, b, c]
        term1 = np.einsum('kab,kcd->kadbc', P_plus.conj(), Q_minus)
        term2 = np.einsum('kab,kcd->kadbc', Q_plus.conj(), P_minus)
    else:  # ph channel
        # Π^{ph,Q}_{μ'ν';μν} = P_{μ'μ}(k+Q/2) Q_{νν'}(k-Q/2) + (P↔Q)
        # row compound = (μ',ν') = (a,d), col compound = (μ,ν) = (b,c)
        term1 = np.einsum('kab,kcd->kadbc', P_plus, Q_minus)
        term2 = np.einsum('kab,kcd->kadbc', Q_plus, P_minus)

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


def nestability_map_2d(eigvecs_2d, channel='ph'):
    """Compute ω̃₀^Q = λ_min(Π^Q) on a 2D BZ grid using periodic roll shifts.

    Evaluates Q on a sub-grid where Q/2 is also a valid grid point, i.e.,
    Q-indices (q1, q2) run over even integers 0, 2, 4, ..., Nk−2,
    giving a (Nk//2) × (Nk//2) nestability map at half the k-grid resolution.

    Args:
        eigvecs_2d: (Nk1, Nk2, Norb, Norb) eigenvectors on uniform BZ grid
        channel:    'ph' (particle-hole, CDW) or 'pp' (particle-particle, SC)

    Returns:
        omega_2d:   (Nk1//2, Nk2//2) nestability map clipped to ≥ 0
        q1_frac:    (Nk1//2,) Q-vector fractional coordinates along B1
        q2_frac:    (Nk2//2,) Q-vector fractional coordinates along B2
    """
    Nk1, Nk2, Norb, _ = eigvecs_2d.shape
    nq1, nq2 = Nk1 // 2, Nk2 // 2

    # Build flat-band projector on 2D grid
    from qgn.geometry import projection_matrix_from_vecs
    P2 = projection_matrix_from_vecs(
        eigvecs_2d.reshape(Nk1 * Nk2, Norb, Norb), [0]
    ).reshape(Nk1, Nk2, Norb, Norb)
    I_orb = np.eye(Norb, dtype=complex)
    Q2 = I_orb[None, None] - P2        # (Nk1, Nk2, Norb, Norb)

    omega_2d = np.zeros((nq1, nq2))

    for iq1 in range(nq1):
        for iq2 in range(nq2):
            dq1 = iq1          # half-Q shift index along axis 0
            dq2 = iq2          # half-Q shift index along axis 1

            P_plus  = np.roll(np.roll(P2,  dq1, axis=0),  dq2, axis=1)
            P_minus = np.roll(np.roll(P2, -dq1, axis=0), -dq2, axis=1)
            Q_plus  = np.roll(np.roll(Q2,  dq1, axis=0),  dq2, axis=1)
            Q_minus = np.roll(np.roll(Q2, -dq1, axis=0), -dq2, axis=1)

            if channel == 'ph':
                term1 = np.einsum('...ab,...cd->...adbc', P_plus,        Q_minus)
                term2 = np.einsum('...ab,...cd->...adbc', Q_plus,        P_minus)
            else:  # pp
                term1 = np.einsum('...ab,...cd->...adbc', P_plus.conj(), Q_minus)
                term2 = np.einsum('...ab,...cd->...adbc', Q_plus.conj(), P_minus)

            Pi_raw = (term1 + term2).mean(axis=(0, 1))   # (Norb,Norb,Norb,Norb)
            Pi = Pi_raw.reshape(Norb ** 2, Norb ** 2)
            Pi = (Pi + Pi.conj().T) / 2
            omega_2d[iq1, iq2] = np.linalg.eigvalsh(Pi).min()

    omega_2d = np.maximum(omega_2d, 0.0)
    q1_frac = np.arange(nq1) / Nk1  * 2   # Q in [0,1) fractional
    q2_frac = np.arange(nq2) / Nk2  * 2
    return omega_2d, q1_frac, q2_frac


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
