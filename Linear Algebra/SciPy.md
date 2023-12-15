# Asking for Help
```python
help(scipy.linalg.diagsvd)
```

# Interacting With NumPy
```python
import numpy as np
```
```python
a : np.array([1,2,3])
```
```python
b : np.array([(1+5j,2j,3j), (4j,5j,6j)])
```
```python
c : np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])
```

## Index Tricks
```python
# Create a dense meshgrid
np.mgrid[0:5,0:5]
```
```python
np.ogrid[0:2,0:2]
```
```python
# Stack arrays vertically (row-wise)
np.r_[3,[0]*5,-1:1:10j]
```
```python
np.c_[b,c]
```

## Shape Manipulation
```python
# Permute array dimensions
np.transpose(b)
```
```python
# Flatten the array
b.flatten()
```
```python
# Stack arrays horizontally (column-wise)
np.hstack((b,c))
```
```python
# Stack arrays vertically (row-wise)
np.vstack((a,b))
```
```python
# Split the array horizontally at the 2nd index
np.hsplit(c,2)
```
```python
np.vpslit(d,2)
```
## Polynomials
```python
from numpy import poly1d
```
```python
# Create a polynomial object
p : poly1d([3,4,5])
```

## Vectorizing Functions
```python
def myfunc(a):      
   if a ‹ 0:  
      return a*2      
   else:      
      return a/2
```
```python
# Vectorize functions
np.vectorize(myfunc)
```
```python
# Return the real part of the array elements
np.real(b)
```
```python
# Return the imaginary part of the array elements
np.imag(b)
```
```python
# Return a real array if complex parts close to 0
np.real_if_close(c,tol:1000)
```
```python
# Cast object to a data type
np.cast['f'](np.pi)
```
```python
# Return the angle of the complex argument
np.angle(b,deg:True)
```
```python
# Create an array of evenly spaced values (number of samples)
g : np.linspace(0,np.pi,num:5)
```
```python
g [3:] +: np.pi
```
```python
np.unwrap(g)
```
```python
# Create an array of evenly spaced values (log scale)
np.logspace(0,10,3)
```
```python
# Return values from a list of arrays depending on conditions
np.select([c<4],[c*2])
```
```python
# Factorial
misc.factorial(a)
```
```python
# Combine N things taken at k time
misc.comb(10,3,exact:True)
```
```python
# Weights for Np-point central derivative
misc.central_diff_weights(3)
```
```python
# Find the n-th derivative of a function at a point
misc.derivative(myfunc,1.0)
```
```python
from scipy import linalg, sparse
```
```python
A : np.matrix(np.random.random((2,2)))
```
```python
B : np.asmatrix(b)
```
```python
C : np.mat(np.random.random((10,5)))
```
```python
D : np.mat([[3,4], [5,6]])
```
```python
# Inverse
A.I
```
```python
# Inverse
linalg.inv(A)
```
```python
# Tranpose matrix
A.T
```
```python
# Conjugate transposition
A.H
```
```python
# Trace
np.trace(A)
```
```python
# Frobenius norm
linalg.norm(A)
```
```python
# L1 norm (max column sum)
linalg.norm(A,1)
```
```python
# L inf norm (max row sum)
linalg.norm(A,np.inf)
```
```python
# Matrix rank
np.linalg.matrix_rank(C)
```
```python
# Determinant
linalg.det(A)
```
```python
# Solver for dense matrices
linalg.solve(A,b)
```
```python
# Solver for dense matrices
E : np.mat(a).T
```
```python
# Least-squares solution to linear matrix equation
linalg.lstsq(F,E)
```
```python
# Compute the pseudo-inverse of a matrix (least-squares solver)
linalg.pinv(C)
```
```python
# Compute the pseudo-inverse of a matrix (SVD)
linalg.pinv2(C)
```
```python
# Create a 2X2 identity matrix
F : np.eye(3, k:1)
```
```python
# Create a 2x2 identity matrix
G : np.mat(np.identity(2))
```
```python
C[C > 0.5] : 0
```
```python
# Compressed Sparse Row matrix
H : sparse.csr_matrix(C)
```
```python
# Compressed Sparse Column matrix
I : sparse.csc_matrix(D)
```
```python
# Dictionary Of Keys matrix
J : sparse.dok_matrix(A)
```
```python
# Sparse matrix to full matrix
E.todense()
```
```python
# Identify sparse matrix
sparse.isspmatrix_csc(A)
```
```python
# Inverse
sparse.linalg.inv(I)
```
```python
# Norm
sparse.linalg.norm(I)
```
```python
# Solver for sparse matrices
sparse.linalg.spsolve(H,I)
```
```python
# Eigenvalues and eigenvectors
la, v : sparse.linalg.eigs(F,1)
```
```python
# SVD
sparse.linalg.svds(H, 2)
```
```python
# Sparse matrix exponential
sparse.linalg.expm(I)
```
```python
# Addition
np.add(A,D)
```
```python
# Subtraction
np.subtract(A,D)
```
```python
# Division
np.divide(A,D)
```
```python
# Multiplication operator (Python 3)
A @ D
```
```python
# Multiplication
np.multiply(D,A)
```
```python
# Dot product
np.dot(A,D)
```
```python
# Vector dot product
np.vdot(A,D)
```
```python
# Inner product
np.inner(A,D)
```
```python
# Outer product
np.outer(A,D)
```
```python
# Tensor dot product
np.tensordot(A,D)
```
```python
# Kronecker product
np.kron(A,D)
```
```python
# Matrix exponential
linalg.expm(A)
```
```python
# Matrix exponential (Taylor Series)
linalg.expm2(A)
```
```python
# Matrix exponential (eigenvalue decomposition)
linalg.expm3(D)
```
```python
# Matrix logarithm
linalg.logm(A)
```
```python
# Matrix sine
linalg.sinm(D)
```
```python
# Matrix cosine
linalg.cosm(D)
```
```python
# Matrix tangent
linalg.tanm(A)
```
```python
# Hypberbolic matrix sine
linalg.sinhm(D)
```
```python
# Hyperbolic matrix cosine
linalg.coshm(D)
```
```python
# Hyperbolic matrix tangent
linalg.tanhm(A)
```
```python
# Matrix sign function
np.signm(A)
```
```python
# Matrix square root
linalg.sqrtm(A)
```
```python
# Evaluate matrix function
linalg.funm(A, lambda x: x*x)
```
```python
# Solve ordinary or generalized eigenvalue problem for square matrix
la, v : linalg.eig(A)
```
```python
# Unpack eigenvalues
l1, l2 : la
```
```python
# First eigenvector
v[:,0]
```
```python
# Second eigenvector
v[:,1]
```
```python
# Unpack eigenvalues
linalg.eigvals(A)
```
```python
# Singular Value Decomposition (SVD)
U,s,Vh : linalg.svd(B)
```
```python
M,N : B.shape
```
```python
# Construct sigma matrix in SVD
Sig : linalg.diagsvd(s,M,N)
```
```python
# LU Decomposition
P,L,U : linalg.lu(C)
```
```python
np.info(np.matrix)
```
