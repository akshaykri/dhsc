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
        self.make_Pauli()

    def make_Pauli(self):
        """define some useful matrices"""
        I = np.array([[1, 0], [0, 1]])
        Sx = np.array([[0, 0.5], [0.5, 0]])
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 1], [0, 0]])
        Sm = np.array([[0, 0], [1, 0]])

        self.Pauli = {"I": I, "Sz": Sz, "Sx": Sx, "Sp": Sp, "Sm": Sm}

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

    def prod_init(self, thetas=None, eiphi=None):
        """Initialize the MPS in a product state (unentangled state) with 
        default: each spin distributed uniformly on the Bloch sphere, with phi=0 or pi"""

        self.left_can = -1 #left canonicalized to this site index ...
        self.right_can = self.N #right canonicalized from this site index ...

        if thetas is None:
            thetas = np.random.rand(self.N) * np.pi

        if eiphi is None:
            eiphi = 1 - 2*np.random.randint(2, size=self.N)

        for s in np.arange(self.N):

            if s == 0:
                Mtemp = np.zeros((self.p, 1, self.chi))       
            elif s == self.N-1:
                Mtemp = np.zeros((self.p, self.chi, 1))
            else:
                Mtemp = np.zeros((self.p, self.chi, self.chi))

            Mtemp[0,0,0] = np.cos(0.5*thetas[s])
            Mtemp[1,0,0] = np.sin(0.5*thetas[s]) * eiphi[s]
            self.M.append(Mtemp)

    def prod_init_z(self):
        """Initialize the MPS with all spins aligned along +z"""
        self.prod_init(thetas=np.zeros(self.N))

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

        flag = 0
        #for left canonicalization
        try:
            I = np.array([[1.0]])
            for cnt in np.arange(0, self.left_can+1):
                I = np.einsum('ij, pia, pjb -> ab', I, self.M[cnt], self.M[cnt], optimize=True)
                assert(np.max(np.abs(I - np.eye(I.shape[0], I.shape[1])))) < 1e-10
        except AssertionError:
            print("Left canonicalization failed at {0:d}th site".format(cnt))
            flag = -1

        #for right canonicalization
        try:
            I = np.array([[1.0]])
            for cnt in np.arange(self.N-1, self.right_can-1, -1):
                I = np.einsum('ij, pai, pbj -> ab', I, self.M[cnt], self.M[cnt], optimize=True)
                assert(np.max(np.abs(I - np.eye(I.shape[0], I.shape[1])))) < 1e-10
        except AssertionError:
            print("Right canonicalization failed at {0:d}th site".format(cnt))
            flag = -1

        #for normalization
        try:
            if self.left_can < self.N - 1:
                I = np.eye(self.M[self.left_can+1].shape[1], self.M[self.left_can+1].shape[1]) # np.array([[1.0]])
                for cnt in np.arange(self.left_can+1, self.right_can, 1):
                    I = np.einsum('ij, pia, pjb -> ab', I, self.M[cnt], self.M[cnt], optimize=True)
                assert(np.abs(np.trace(I) - 1)) < 1e-10

        except AssertionError:
            print("Normalization failed")
            flag = -1

        if flag == 0:
            print("Verified left canonicalized to the {0:d}th site and right canonicalized from the {1:d}th site and normalized"
            .format(self.left_can, self.right_can))


    def normalize(self, n):
        """Normalize the MPS"""
        
        self.left_canonicalize(n-1)
        self.right_canonicalize(n+1)
        norm = np.sum(self.M[n] * self.M[n].conj())
        self.M[n] = self.M[n] / np.sqrt(norm)


    def get_EE(self, n=None):
        """Calculate the entanglement entropy of the spin chain between sites n and n+1"""
        if n is None:
            n = self.N/2 - 1
        self.normalize(n)
        d, chiL, chiR = self.M[n].shape # original shape of tensor
        _, S, _ = np.linalg.svd(self.M[n].reshape(d*chiL, chiR))
        assert np.abs(np.sum(np.abs(S)**2) - 1) < 1e-10
        return -np.sum(np.abs(S[np.abs(S)>0])**2 * np.log(np.abs(S[np.abs(S)>0])**2))

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
        """return the quadratic form of an MPS between a generic MPO"""
        try:
            assert self.N == mpoobj.N
            C = np.array([[[1]]])
            for c in np.arange(self.N):
                C = np.einsum('abc, sad, best, tcf -> def', C, self.M[c], mpoobj.W[c], self.M[c],
                    optimize=True)
            return C[0, 0, 0]

        except AssertionError:
            print("Number of tensors not compatible")

    def apply_site_operator(self, n=None, op=None, normalize=False):
        """apply a single 2 x 2 pauli type operator to site n"""

        if n is None:
            n = self.N/2

        if op is None:
            op = np.eye(self.p)

        self.M[n] = np.einsum('il, ljk -> ijk', op, self.M[n], optimize=True)

        if normalize:
            self.left_canonicalize(n-1)
            self.right_canonicalize(n+1)
            norm = np.sum(self.M[n] ** 2)
            self.M[n] = self.M[n] / np.sqrt(norm)


    def measure_site_operator(self, n=None, op=None):
        """measure the expectation value of a single 2 x 2 pauli type operator at site n"""

        if n is None:
            n = self.N/2

        if op is None:
            op = np.eye(self.p)

        self.left_canonicalize(n-1)
        self.right_canonicalize(n+1)

        Braket = np.einsum('sij, tij, st -> ', self.M[n], self.M[n], op, optimize=True)
        return Braket

    def TDVP_evolve(self, mpoobj, dt=1e-3, t_max=1):
        """the magnum opus: Appendix B of Haegeman et al PRB 94, 165116 (2016)"""

        self.right_canonicalize()
        self.normalize()

        nsweep = int(t_max / dt)
        #TAKE COMPLEX CONJUGATES

        R = []
        R.append(np.array([[[1]]])) # R_{N-1} is trivial
        for c in np.arange(self.N-1, 0, -1): # pre-pend R_{N-2} thru R_0
            R.insert(0, np.einsum('abc, ebst, sda, tfc -> def', 
                R[0], mpoobj.W[c], self.M[c], self.M[c], optimize=True))

        for it in np.arange(nsweep):
            #sweep right

            L = []
            L.append(np.array([[[1]]])) # L_0 is trivial
            for lc in np.arange(self.N-1):
                #this is H(n)
                OP = np.einsum('abd, bfst, efh -> tdhsae', L[lc], mpoobj.W[lc], R[lc], optimize=True)
                L1, L2, L3, L4, L5, L6 = OP.shape

                #step 1a
                E = scipy.sparse.linalg.expm( -0.5j * dt * scipy.sparse.csr_matrix(
                    OP.reshape(L1*L2*L3, L4*L5*L6)) )
                Mn = E.dot(self.M[lc].reshape(L1*L2*L3))

                #step 1b
                q, r = np.linalg.qr(Mn.reshape(L1*L2, L3))
                self.M[lc] = q.reshape(L1, L2, L3)

                #step 1c
                #this is K(n)
                OQ = np.einsum('tdhsae, tdp, saq -> phqe', OP, self.M[lc], self.M[lc], optimize=True)
                K1, K2, K3, K4 = OQ.shape
                F = scipy.sparse.linalg.expm( +0.5j * dt * scipy.sparse.csr_matrix(
                    OQ.reshape(K1*K2, K3*K4)) )
                rn = F.dot(r.reshape(K1*K2))

                #step 1d
                self.M[lc+1] = np.einsum('il, jlk -> jik', rn, self.M[lc+1], optimize=True)
                self.left_can = lc
                if lc > self.right_can-2: self.right_can = np.minimum(lc+2, self.N)

                L.append(np.einsum('abc, best, sad, tcf -> def', 
                    L[lc], mpoobj.W[lc], self.M[lc], self.M[lc], optimize=True)) # append L_c for lc=1 thru N-1

            #step 2
            HN = np.einsum('abd, bfst, efh -> tdhsae', L[self.N-1], mpoobj.W[self.N-1], R[self.N-1], optimize=True)
            L1, L2, L3, L4, L5, L6 = HN.shape

            E = scipy.sparse.linalg.expm( -1j * dt * scipy.sparse.csr_matrix(
                    HN.reshape(L1*L2*L3, L4*L5*L6)) )
            self.M[self.N-1] = E.dot(self.M[self.N-1].reshape(L1*L2*L3))

            #step 3 left sweep
            R = []
            R.append(np.array([[[1]]]))
            for rc in np.arange(self.N-2, -1, -1):
                
                #step 3a
                L4, L5, L6 = self.M[rc+1].shape
                q, r = np.linalg.qr(self.M[rc+1].transpose(0,2,1).reshape(L4*L6, L5))
                
                #step 3b
                #this is K(n)
                OQ = np.einsum(' -> phqe', L[rc+1], R[0], mpoobj.W[0], self.M[lc], self.M[lc], optimize=True)
                K1, K2, K3, K4 = OQ.shape
                F = scipy.sparse.linalg.expm( +0.5j * dt * scipy.sparse.csr_matrix(
                    OQ.reshape(K1*K2, K3*K4)) )
                rn = F.dot(r.reshape(K1*K2))
                MGS.M[lc] = q.reshape(L4, L6, L5).transpose((0, 2, 1))

