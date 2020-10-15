import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
A100 = random.randn(100,100) + 1j*random.randn(100,100)
Q100, R100 = np.linalg.qr(A100)
b100 = random.randn(100,100) + 1j*random.randn(100,100)
A200 = random.randn(200,200) + 1j*random.randn(200,200)
Q200, R200 = np.linalg.qr(A200)
b200 = random.randn(200,200) + 1j*random.randn(200,200)
A400 = random.randn(400,400) + 1j*random.randn(400,400)
Q400, R400 = np.linalg.qr(A400)
b400 = random.randn(400,400) + 1j*random.randn(400,400)


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    (m, n) = Q.shape
    r = v
    u = np.zeros(n)

    for i in range(n):
        u[i] = np.dot(Q[:, i].T, v)
        r = r - u[i] * Q[:, i]
    
    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    
    Q_conj = Q.conj().T
    x = np.dot(Q_conj, b)
    
    return x

def timeable_solveQ100():
    """
    Doing a solveQ example with m = 100 that we can
    pass to timeit.
    """
    
    b = solveQ(Q100, b100)

def timeable_solveQ200():
    """
    Doing a solveQ example with m = 200 that we can
    pass to timeit.
    """
    
    b = solveQ(Q200, b200)

def timeable_solveQ400():
    """
    Doing a solveQ example with m = 400 that we can
    pass to timeit.
    """
    
    b = solveQ(Q400, b400)
    
def timeable_np100():
    """
    Doing a inbuilt example with m = 100 that we can
    pass to timeit.
    """
    
    b = np.linalg.solve(Q100, b100)

def timeable_np200():
    """
    Doing a inbuilt example with m = 200 that we can
    pass to timeit.
    """
    
    b = np.linalg.solve(Q200, b200)

def timeable_np400():
    """
    Doing a inbuilt example with m = 400 that we can
    pass to timeit.
    """
    
    b = np.linalg.solve(Q400, b400)
    
def time_solveQ():
    """
    Get some timings for solveQ.
    """

    print("Timing for solveQ for m = 100")
    print(timeit.Timer(timeable_solveQ100).timeit(number=1))
    print("Timing for solveQ for m = 200")
    print(timeit.Timer(timeable_solveQ200).timeit(number=1))
    print("Timing for solveQ for m = 400")
    print(timeit.Timer(timeable_solveQ400).timeit(number=1))
    print("Timing for inbuilt for m = 100")
    print(timeit.Timer(timeable_np100).timeit(number=1))
    print("Timing for inbuilt for m = 200")
    print(timeit.Timer(timeable_np200).timeit(number=1))
    print("Timing for inbuilt for m = 400")
    print(timeit.Timer(timeable_np400).timeit(number=1))

def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    raise NotImplementedError

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an lxm-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """

    raise NotImplementedError

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    raise NotImplementedError

    return Q, R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, producing

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    raise NotImplementedError

    return Q, R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    raise NotImplementedError

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.eye(n)
    for i in range(1,m):
        Rk = GS_modified_get_R(A, i)
        np.dot(A, Rk, out=A)
        np.dot(R, Rk, out=R)
    R = np.linalg.inv(R)
    return A, R
