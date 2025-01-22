import numpy as np
from tqdm import tqdm
from scipy.linalg import eig, inv

def cheb_nodes(N):
    return np.sin(np.pi*np.arange(N-1, -N, -2).reshape(N, 1)/(2*(N-1)))


def linearCompanionMatrices(M, A2, A1, A0):
    """
    Parameters
    ----------
        M: int
            size of the problem. Uses 2*M by 2*M matrices
        Ai: arrays of (M, M)
            Matrices set for the problem 
                (l^2 A_2 + l A_1 + A_0)u = 0

    Returns
    -------
        Ap, Bp: arrays of size  (2*M, 2*M)
            Matrices of the GEP 
                    (Ap - l Bp)U = 0,  
            U = (l u, u)
    """
    
    IIM = np.eye(M)
    # Create the companion matrices
    Ap = np.zeros((2*M, 2*M), dtype=complex)
    Ap[:M, :M] = -A1;     Ap[:M, M:2*M] = -A0;    Ap[M:2*M, :M] = IIM
    Bp = np.zeros((2*M, 2*M), dtype=complex)
    Bp[:M, :M] = A2;      Bp[M:2*M, M:2*M] = IIM

    return Ap, Bp


def chebdif(N, M):
    """
    The function chebdif(N,M) computes the differentiation
    matrices D1, D2, ..., DM on Chebyshev nodes.

    Input:
    N:        Size of differentiation matrix.
    M:        Number of derivatives required (integer).
    Note:     0 < M <= N-1.

    Output:
    x:        Chebyshev points
    DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.

    The code implements two strategies for enhanced
    accuracy suggested by W. Don and S. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The two strategies are (a) the use of trigonometric
    identities to avoid the computation of differences
    x(k)-x(j) and (b) the use of the "flipping trick"
    which is necessary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    Note added May 2003:  It may, in fact, be slightly better not to
    implement the strategies (a) and (b).   Please consult the following
    paper for details:   "Spectral Differencing with a Twist", by
    R. Baltensperger and M.R. Trummer, to appear in SIAM J. Sci. Comp.

    J.A.C. Weideman, S.C. Reddy 1998.  Help notes modified by
    JACW, May 2003."""
    
    from scipy.linalg import toeplitz

    I = np.eye(N, dtype=bool)  # Identity boolean matrix.

    n1 = int(np.floor(N/2))  # Indice used for flipping trick.
    n2 = int(np.ceil(N/2))  # Indice used for flipping trick.

    k = np.arange(N).reshape(N, 1)  # Compute theta vector.
    th = k*np.pi/(N-1)

    x = cheb_nodes(N)
    # np.sin(np.pi*np.arange(N-1, -N, -2).reshape(N, 1)/(2*(N-1)))  # Compute Chebyshev points.

    T = np.repeat(th/2, N, axis=1)
    DX = 2 * np.sin(np.transpose(T) + T) * np.sin(np.transpose(T) - T)  # Trigonometric identity.
    DX = np.concatenate((DX[0:n1, :], -np.rot90(DX[0:n2, :], 2)), axis=0)  # Flipping trick.
    DX[I] = 1  # Put 1's on the main diagonal of DX.

    C = toeplitz(np.power(-1., k))  # C is the matrix with entries c(k)/c(j)
    C[0, :] = C[0, :] * 2
    C[-1, :] = C[-1, :] * 2
    C[:, 0] = C[:, 0] / 2
    C[:, -1] = C[:, -1] / 2

    Z = 1/DX  # Z contains entries 1/(x(k)-x(j))
    Z[I] = 0  # with zeros on the diagonal.

    D = np.eye(N)  # D contains diff. matrices.
    DM = np.zeros((N, N, M))
    for ell in range(M):
        D = (ell+1) * Z * (C * np.repeat(np.diag(D).reshape(N, 1), N, axis=1) - D)  # Off-diagonals
        D[I] = -np.sum(np.transpose(D), axis=0)  # Correct main diagonal of D
        DM[:, :, ell] = D  # Store current D in DM

    return x, DM


def lamb(freqs, materials_properties):
    M = [materials_properties['elastic']['M']]
    h = [materials_properties['elastic']['h']]
    pb_size = 2*M[0]
    elastic = materials_properties['elastic']['params']

    rhos = elastic['rho']
    ld   = elastic['ld'];    
    mu = elastic['mu']
    # print(rhos, ld, mu)

    xi_cheby = []; T2 = []; T22 = []; II = []
    for i in range(len(M)):
        xi, TT = chebdif(M[i], 2)
        xi_cheby.append(xi)
        T2.append(TT[:, :, 0]*2*h[i])
        T22.append(TT[:, :, 1]*4)
        II.append(np.eye(M[i])*h[i]**2)

    A2 = np.zeros((pb_size, pb_size), dtype=complex)
    A1 = np.zeros((pb_size, pb_size), dtype=complex)
    A0 = np.zeros((pb_size, pb_size), dtype=complex)

    eigval =   np.zeros((len(freqs), 2*pb_size), dtype=complex)
    eigvec =   np.zeros((len(freqs), 2*pb_size, 2*pb_size), dtype=complex)

    u1 = slice(0, M[0]); u2 = slice(M[0], 2*M[0])
    # Boundary conditions indices
    BC_BOT = [
          M[0] - 1, 
        2*M[0] - 1, 
    ]

    BC_TOP = [ 
        0, 
        M[0], 
    ]

    BCS = [*BC_BOT, *BC_TOP]
    EQROWS = [slice(i*M[0], (i+1)*M[0]) for i in range(2)]
    top = -1; bot = 0

    for ii, f in enumerate(freqs):

        om = 2*np.pi*f

        # Equations of motion : Elastic plate
        A0[EQROWS[0], u1] = rhos/mu*om**2*II[0] +             T22[0]
        A2[EQROWS[0], u1] =                     -1*(ld/mu + 2)*II[0]
        A1[EQROWS[0], u2] =                     1j*(ld/mu + 1)*T2[0]
        A1[EQROWS[1], u1] =                     1j*(ld/mu + 1)*T2[0]
        A0[EQROWS[1], u2] = rhos/mu*om**2*II[0] + (ld/mu + 2)*T22[0]
        A2[EQROWS[1], u2] =                                   -II[0]

        A2[BCS, :] = 0;     A1[BCS, :] = 0;     A0[BCS, :] = 0

        A0[BC_TOP[0], u1] =                      T2[0][top, :]
        A1[BC_TOP[0], u2] =                   1j*II[0][top, :]
        A1[BC_TOP[1], u1] =             1j*ld/mu*II[0][top, :]
        A0[BC_TOP[1], u2] =          (ld/mu + 2)*T2[0][top, :]

        A0[BC_BOT[0], u1] =                      T2[0][bot, :]
        A1[BC_BOT[0], u2] =                   1j*II[0][bot, :]
        A1[BC_BOT[1], u1] =             1j*ld/mu*II[0][bot, :]
        A0[BC_BOT[1], u2] =          (ld/mu + 2)*T2[0][bot, :]

        Ap, Bp = linearCompanionMatrices(pb_size, A2, A1, A0)

        eigval[ii, :], eigvec[ii, :, :] = eig(Ap, Bp)    
    
    return eigval, eigvec
