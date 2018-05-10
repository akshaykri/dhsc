import numpy as np
import time
import scipy
import scipy.misc
import scipy.sparse
import matplotlib.pyplot as plt
from mps import MPS

class MPO(object):
    """Encapsulates all kinds of useful information about a matrix product operator"""

    def __init__(self, params={}):
        self.N = params.get("N", 20) #number of sites
        self.p = params.get("p", 2) #dimension of local Hilbert space
        self.chio = params.get("chio", 5) #bond dimension
        self.W = []

        # For the most generic XYZ Hamiltonian with X and Z fields
        self.Jx = params.get("Jx", np.ones(self.N-1))
        self.Jy = params.get("Jy", np.ones(self.N-1))
        self.Jz = params.get("Jz", np.ones(self.N-1))
        self.hx = params.get("hx", np.zeros(self.N))
        self.hz = params.get("hz", np.zeros(self.N))

    
    def make_MPO(self):
        """create the MPO"""
        (I, Sx, Sz, Sp, Sm) = self.make_Pauli()
        for s in np.arange(self.N):
            if s == 0:
                W0 = np.zeros((1, self.chio, self.p, self.p))
                W0[0,:,:,:] = np.array([I, Sp, Sm, Sz, self.hx[s]*Sx + self.hz[s]*Sz])
                self.W.append(W0)
            elif s == self.N-1:
                W0 = np.zeros((self.chio, 1, self.p, self.p))
                W0[:,0,:,:] = np.array([self.hx[s]*Sx + self.hz[s]*Sz,
                    0.25*(self.Jx[s-1] + self.Jy[s-1])*Sm + 
                    0.25*(self.Jx[s-1] - self.Jy[s-1])*Sp,
                    0.25*(self.Jx[s-1] + self.Jy[s-1])*Sp + 
                    0.25*(self.Jx[s-1] - self.Jy[s-1])*Sm,
                    self.Jz[s-1]*Sz, I])
                self.W.append(W0)
            else:
                W0 = np.zeros((self.chio, self.chio, self.p, self.p))
                W0[0,:,:,:] = np.array([I, Sp, Sm, Sz, self.hx[s]*Sx + self.hz[s]*Sz])
                W0[:,-1,:,:] = np.array([self.hx[s]*Sx + self.hz[s]*Sz,
                    0.25*(self.Jx[s-1] + self.Jy[s-1])*Sm + 
                    0.25*(self.Jx[s-1] - self.Jy[s-1])*Sp,
                    0.25*(self.Jx[s-1] + self.Jy[s-1])*Sp + 
                    0.25*(self.Jx[s-1] - self.Jy[s-1])*Sm,
                    self.Jz[s-1]*Sz, I])
                self.W.append(W0)


    def make_Pauli(self):
        """define some useful matrices"""
        I = np.array([[1, 0], [0, 1]])
        Sx = np.array([[0, 0.5], [0.5, 0]])
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 1], [0, 0]])
        Sm = np.array([[0, 0], [1, 0]])

        return (I, Sx, Sz, Sp, Sm)


    def act_on_MPS(self, mpsobj):
        """act this MPO on an MPS to return another MPS object"""
        try:
            assert self.N == mpoobj.N
            mpsret = MPS({'N': mpsobj.N, 'p': mpsobj.p, 'chi': mpsobj.chi * self.chio})
            for c in np.arange(self.N):
                P = np.einsum('cdst, tab -> scadb', self.W[c], mpsobj.M[c], optimize='True')
                L1, L2, L3, L4, L5 = P.shape
                mpsret.M.append(P.reshape(L1, L2*L3, L4*L5))

            mpsret.left_can = -1
            mpsret.right_can = self.N
            return mpsret

        except AssertionError:
            print("Number of tensors not compatible")

    def get_GS(self, params={'chi': 10}):
        """find the ground state of the MPO with the given parameters"""

        MGS = MPS({'N': self.N, 'p': self.p, 'chi': params['chi']})
        MGS.rand_init()
        MGS.left_canonialize()
        MGS.right_canonicalize()

        R = []
        R.append(np.array([[[1]]]))
        for c in np.arange(self.N-1, 0, -1):
            R.insert(0, np.einsum('abc, ebst, sda, tfc -> def', 
                R[0], self.W[c], MGS[c], MGS[c], optimize='True'))

        L = []
        L.append(np.array([[[1]]]))
        for it in np.arange(nsweep):

            #sweep right
            for lc in np.arange(self.N):
                OP = np.einsum('abd, bfst, efh -> tdhsae', L[lc], self.W[lc], R[lc], optimize='True')
                L1, L2, L3, L4, L5, L6 = OP.shape
                OP.reshape(L1*L2*L3, L4*L5*L6)
                lamb, mat = scipy.sparse.linalg.eigsh(..., which='SA', k=1)
                # convert to sparse
                # take first eigenvector only
                Q, R = np.linalg.qr(mat.reshape(...))
                MGS.M[lc] = Q.reshape(L4, L5, L6)
                if lc < self.N - 1:
                    MGS.M[lc+1] = np.einsum('il, jlk -> jik', R, MGS.M[s+1], optimize=True)
                L.append(np.einsum('abc, best, sad, tcf -> def', 
                    L[lc], self.W[c], MGS[c], MGS[c], optimize='True'))

            #sweep left


