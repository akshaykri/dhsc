import numpy as np
import time
import scipy
import scipy.misc
import scipy.sparse
import matplotlib.pyplot as plt

class MPS(object):
    """Encapsulates all kinds of information about a matrix-product state"""

    def __init__(self, params={}):
        self.N = params.get("N", 20) #number of sites
        self.p = params.get("p", 2) #dimension of local Hilbert space
        self.chi = params.get("chi", 10) #bond dimension
        self.rand_seed = params.get("rand_seed", 1) #seed for random number generator
        #self.normalized = False
        #self.canonical = False
        self.M = [] # list of MPS tensors

    def rand_init(self):
        """Initialize the MPS with random tensors of the appropriate dimension"""

        self.left_can = -1 #left canonicalized to this site index ...
        self.right_can = self.N #right canonicalized from this site index ...

        np.random.seed(self.rand_seed)
        for s in np.arange(self.N):
            if s == 0:
                self.M.append(np.random.randn(self.p, 1, self.chi))
            elif s == self.N-1:
                self.M.append(np.random.randn(self.p, self.chi, 1))
            else:
                self.M.append(np.random.randn(self.p, self.chi, self.chi))

    def left_canonicalize(self, Lmax=None):
        """Left canonicalize the MPS up to (and including) the Lmax^th site"""
        if Lmax is None: Lmax = self.N - 1
        for s in np.arange(self.left_can+1, Lmax+1):
            d, chiL, chiR = self.M[s].shape # original shape of tensor
            Q, R = np.linalg.qr(self.M[s].reshape(d*chiL, chiR))
            self.M[s] = Q.reshape(d, chiL, Q.shape[1])
            if s < self.N - 1:
                self.M[s+1] = np.einsum('il, jlk -> jik', R, self.M[s+1], optimize=True)
            self.left_can = s
            if s > self.right_can-2: self.right_can = np.minimum(s+2, self.N)

    def right_canonicalize(self, Lmin=None):
        """Right canonicalize the MPS from (and including) the Lmin^th site"""
        if Lmin is None: Lmin = 0
        for s in np.arange(self.right_can-1, Lmin-1, -1):
            d, chiL, chiR = self.M[s].shape # original shape of tensor
            Q, R = np.linalg.qr((self.M[s].transpose((0, 2, 1)).
                reshape(d*chiR, chiL)))
            self.M[s] = Q.reshape(d, chiR, Q.shape[1]).transpose((0, 2, 1))
            if s > 0:
                self.M[s-1] = np.einsum('il, jkl -> jki', R, self.M[s-1], optimize=True)
            self.right_can = s
            if s < self.left_can+2: self.left_can = np.maximum(s-2, -1)

    def check_canonicalization(self):
        """Check if orthonormality relations for tensors hold up"""

        #for left canonicalization
        try:
            I = np.array([[1.0]])
            for cnt in np.arange(0, self.left_can+1):
                I = np.einsum('ij, pia, pjb -> ab', I, self.M[cnt], self.M[cnt], optimize=True)
                assert(np.max(np.abs(I - np.eye(I.shape[0], I.shape[1])))) < 1e-10
        except AssertionError:
            print("Left canonicalization failed at {0:d}th site".format(cnt))
            return

        #for right canonicalization
        try:
            I = np.array([[1.0]])
            for cnt in np.arange(self.N-1, self.right_can-1, -1):
                I = np.einsum('ij, pai, pbj -> ab', I, self.M[cnt], self.M[cnt], optimize=True)
                assert(np.max(np.abs(I - np.eye(I.shape[0], I.shape[1])))) < 1e-10
        except AssertionError:
            print("Right canonicalization failed at {0:d}th site".format(cnt))
            return

        print("Verified left canonicalized to the {0:d}th site and right canonicalized from the {1:d}th site"
            .format(self.left_can, self.right_can))

    def get_EE(self, n=None):
        """Calculate the entanglement entropy of the spin chain between sites n and n+1"""
        if n is None:
            n = self.N/2 - 1

        self.left_canonicalize(n-1)
        self.right_canonicalize(n+1)
        d, chiL, chiR = self.M[n].shape # original shape of tensor
        _, S, _ = np.linalg.svd(self.M[n].reshape(d*chiL, chiR))
        assert np.abs(np.sum(np.abs(S)**2) - 1) < 1e-10
        return -np.sum(np.abs(S)**2 * np.log(np.abs(S)**2))

    def inner_product(self, mpsobj):
        """return a scalar result for the inner product between this MPS and another"""
        try:
            assert self.N == mpsobj.N
            C = np.array([[1]])
            for c in np.arange(self.N):
                C = np.einsum('ab, sap, sbq -> pq', C, self.M[c], mpsobj.M[c], optimize=True)
            return C[0,0]

        except AssertionError:
            print("Number of tensors not compatible")
        
    def expectation_val(self, mpoobj):
        """return the quadratic form of an MPS between an MPO"""
        try:
            assert self.N == mpoobj.N
            C = np.array([[[1]]])
            for c in np.arange(self.N):
                C = np.einsum('abc, sad, best, tcf -> def', C, self.M[c], mpoobj.W[c], self.M[c],
                    optimize=True)
            return C[0, 0, 0]

        except AssertionError:
            print("Number of tensors not compatible")

        