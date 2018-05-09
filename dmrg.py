import numpy as np
import time
import scipy
import scipy.misc
import scipy.sparse
import matplotlib.pyplot as plt

class DMRG(object):
    """Encapsulates all useful information about a spin half disordered AFM Heisenberg chain
    H = \sum_i S_i \dot S_{i+1} + h_i S_i^z
    """

    def __init__(self, params={}):
        # read parameters and initialize
        self.N = params.get("N", 10) # number of sites
        self.W = params.get("W", 5) #disorder strength -W to W
        self.BC = params.get("BC", 'open') #boundary condition
        self.Sz = params.get("Sz", 0) # which total spin sector
        self.nUp = int(0.5*self.N+self.Sz) # number of up spins
        self.dimH = int(scipy.misc.comb(self.N, self.nUp)) # dimension of Hilbert space
        self.matrixType = params.get("matrixType", 'dense')
        self.seed = params.get("seed", 1)
        np.random.seed(self.seed)
        self.h_dis = params.get("h_dis", self.W*2*(np.random.rand(self.N)-0.5))
        self.params = {"N": self.N, "W": self.W, "BC": self.BC, "Sz": self.Sz, "matrixType": 
            self.matrixType, "seed": self.seed, "h_dis" : self.h_dis}
        self.chi = 10 # Bond dimension
        self.nsweep = 5 # number of sweeps

    def make_dhsc_MPO(self):
        #create a MPO representation of the disordered Heisenberg spin chain
        self.H = [];
        f = self.h_dis # random fields
        # define the Pauli-like 2x2 matrices 
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 1],[0, 0]])
        Sm = np.array([[0, 0],[1, 0]])
        SI = np.eye(2)

        self.H.append(np.concatenate((f[0]*Sz[:,:,np.newaxis], 0.5*Sm[:,:,np.newaxis], 
            0.5*Sp[:,:,np.newaxis], Sz[:,:,np.newaxis], SI[:,:,np.newaxis]), axis=2)) # row MPO for the first site

        for cnt in (np.arange(self.N-2)+1): # matrix MPOs for middle site
            Hi = np.zeros((2,2,5,5))
            Hi[:,:,0,0] = SI
            Hi[:,:,1,0] = Sp
            Hi[:,:,2,0] = Sm
            Hi[:,:,3,0] = Sz
            Hi[:,:,4,0] = f[cnt]*Sz
            Hi[:,:,4,1] = 0.5*Sm
            Hi[:,:,4,2] = 0.5*Sp
            Hi[:,:,4,3] = Sz
            Hi[:,:,4,4] = SI
            self.H.append(Hi)

        self.H.append(np.concatenate((SI[:,:,np.newaxis], Sp[:,:,np.newaxis],
            Sm[:,:,np.newaxis], Sz[:,:,np.newaxis], f[-1]*Sz[:,:,np.newaxis]), axis=2)) # column MPO for the last site

    def make_MPS(self):
        self.M = []
        M.append(np.zeros((self.chi,2))) #first
        for cnt in (np.arange(self.N-2)+1): # matrix MPSs for middle site
            M.append(np.zeros(self.chi, self.chi, 2))
        M.append(np.zeros((self.chi,2))) #last

    def find_GS(self):
        return 0

    def t_evolve(self):
        return 0

class MPS(object):
    """Encapsulates all kinds of information about a matrix-product state"""

    def __init__(self, params={}):
        self.N = params.get("N", 20) #number of sites
        self.p = params.get("N", 2) #dimension of local Hilbert space
        self.chi = params.get("chi", 10) #bond dimension
        self.rand_seed = params.get("rand_seed", 1) #seed for random number generator
        self.normalized = False
        self.canonical = False
        #self.M = np.zeros((self.N-2, self.chi, self.chi, self.p)) #MPS tensors for middle states
        #self.ML = np.zeros((self.chi, self.p))
        #self.MR = np.zeros((self.chi, self.p))
        self.M = []
        

        self.rand_init()
        self.left_can = -1
        self.right_can = self.N

    def rand_init(self):
        """Initialize the MPS with random tensors of the appropriate dimension"""
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
                I = np.einsum('ij, pia, pjb -> ab', I, self.M[cnt], self.M[cnt])
                assert(np.max(np.abs(I - np.eye(I.shape[0], I.shape[1])))) < 1e-10
        except AssertionError:
            print("Left canonicalization failed at {0:d}th site".format(cnt))
            return

        #for right canonicalization
        try:
            I = np.array([[1.0]])
            for cnt in np.arange(self.N-1, self.right_can-1, -1):
                I = np.einsum('ij, pai, pbj -> ab', I, self.M[cnt], self.M[cnt])
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


    def make_canonical(self):
        #self.canonical = True
        return 0

    def conjugate(self):
        #return a bra version of the ket
        MPSbra = MPS() 

    def normalize(self):
        #self.normalized = True
        norm = self.inner_product(self)

        return 0

    def inner_product(self, bra):
        #return a scalar < bra | self >
        
        return 0

class MPO(object):
    """Encapsulates all kinds of useful information about a matrix product operator"""

    def __init__(self, params={}):
            self.N = params.get("N", 20) #number of sites
            self.p = params.get("N", 2) #dimension of local Hilbert space
            self.chi = params.get("chi", 10) #bond dimension
            self.normalized = False
            self.canonical = False
    
    def act_on_MPS(self, MPSobj):
        return 0

