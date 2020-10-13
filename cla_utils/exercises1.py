import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)
u0 = random.randn(400)
v0 = random.randn(400)

def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    (m, n) = A.shape
    b = np.zeros(m)
    
    for i in range(m):
        for j in range(n):
            b[i] += A[i, j] * x[j]
            
    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    (m, n) = A.shape
    b = np.zeros(m)
    i = 0
    for j in x:
        b += j * A[:, i]
        i += 1
    return b


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """
    
    b = column_matvec(A0, x0) # noqa
    
def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*u2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u1: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    
    Bt = np.array([u1, u2])
    B = np.transpose(Bt)
    Cconj = np.array([v1, v2])
    C = Cconj.conj()

    A = B.dot(C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """

    (m, ) = np.shape(u)
    I = np.identity(m)
    vconj = v.conj()
    a = -1/(1 + vconj.dot(u))
    Ainv = I + a * (np.outer(u, vconj))

    return Ainv

def timeable_rank1pert_inv():
    """
    Doing rank1 invert example with the rank1pert_inv that we can pass to timeit
    """
    
    b = rank1pert_inv(u0, v0)
    
def timeable_inbuilt_inv():
    """
    Doing rank1 invert example with the built in inverse that we can pass to timeit
    """
    
    (m, ) = np.shape(u0)
    I = np.identity(m)
    v0conj = v0.conj()
    A = I + np.outer(u0, v0conj)
    b = np.linalg.inv(A)
    
def time_inv():
    """
    Get some timings for inverses
    """
    
    print("Timing for rank1pert_inv")
    print(timeit.Timer(timeable_rank1pert_inv).timeit(number=1))
    print("Timing for inbuilt_inv")
    print(timeit.Timer(timeable_inbuilt_inv).timeit(number=1))

def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i<=j and Ahat[i,j] = C[i,j] for i>j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """

    B = np.triu(Ahat) + np.transpose(np.triu(Ahat, 1))
    C = np.tril(Ahat, 0) - np.transpose(np.tril(Ahat, 0))
    
    (m, ) = np.shape(xi)
    
    zr = np.zeros(m)
    zi = np.zeros(m)
    
    for j in range(m):
        zr += xr[j]*B[:, j] - xi[j]*C[:, j]
        zi += xr[j]*C[:, j] + xi[j]*B[:, j]
        
    return zr, zi
