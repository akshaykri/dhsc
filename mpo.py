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
            assert self.N == mpsobj.N
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

    def get_GS(self, params={'chi': 10}, nsweep=10):
        """find the ground state of the MPO with the given parameters"""

        MGS = MPS({'N': self.N, 'p': self.p, 'chi': params['chi']})
        MGS.rand_init()
        MGS.left_canonicalize()
        MGS.right_canonicalize()

        R = []
        R.append(np.array([[[1]]])) # R_{N-1} is trivial
        for c in np.arange(self.N-1, 0, -1): # pre-pend R_{N-2} thru R_0
            R.insert(0, np.einsum('abc, ebst, sda, tfc -> def', 
                R[0], self.W[c], MGS.M[c], MGS.M[c], optimize=True))

        lamball = []
        
        for it in np.arange(nsweep):

            #sweep right
            L = [] 
            L.append(np.array([[[1]]])) # L_0 is trivial
            for lc in np.arange(self.N-1):
                OP = np.einsum('abd, bfst, efh -> tdhsae', L[lc], self.W[lc], R[lc], optimize=True)
                L1, L2, L3, L4, L5, L6 = OP.shape
                
                lamb, mat = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(
                    OP.reshape(L1*L2*L3, L4*L5*L6)), which='SA', k=1,
                    v0=MGS.M[lc].reshape(L4*L5*L6) )

                q, r = np.linalg.qr(mat[:,0].reshape(L4*L5, L6))
                MGS.M[lc] = q.reshape(L4, L5, L6)

                if lc < self.N - 1: # this condition is always true by the limits of the for loop
                    MGS.M[lc+1] = np.einsum('il, jlk -> jik', r, MGS.M[lc+1], optimize=True)
                MGS.left_can = lc
                if lc > MGS.right_can-2: MGS.right_can = np.minimum(lc+2, MGS.N)

                L.append(np.einsum('abc, best, sad, tcf -> def', 
                    L[lc], self.W[lc], MGS.M[lc], MGS.M[lc], optimize=True)) # append L_c for lc=1 thru N-1
                lamball.append(lamb[0])

            #sweep left
            R = []
            R.append(np.array([[[1]]])) # R_{N-1} is trivial
            for lc in np.arange(self.N-1, 0, -1):
                OP = np.einsum('abd, bfst, efh -> tdhsae', L[lc], self.W[lc], R[0], optimize=True)
                L1, L2, L3, L4, L5, L6 = OP.shape #L4 = d, L5 = chiL, L6 = chiR
                
                lamb, mat = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(
                    OP.reshape(L1*L2*L3, L4*L5*L6)), which='SA', k=1,
                    v0=MGS.M[lc].reshape(L4*L5*L6) )

                q, r = np.linalg.qr(mat[:,0].reshape(L4, L5, L6).transpose(0,2,1).reshape(L4*L6, L5))
                MGS.M[lc] = q.reshape(L4, L6, L5).transpose((0, 2, 1))
                if lc > 0: #always satisfied because of how the for loop is defined
                    MGS.M[lc-1] = np.einsum('il, jkl -> jki', r, MGS.M[lc-1], optimize=True)
                MGS.right_can = lc
                if lc < MGS.left_can+2: MGS.left_can = np.maximum(lc-2, -1)

                R.insert(0, np.einsum('abc, ebst, sda, tfc -> def', 
                    R[0], self.W[lc], MGS.M[lc], MGS.M[lc], optimize=True)) # append R_c for lc=N-2 thru 0

                lamball.append(lamb[0])

        return (MGS, lamball)
