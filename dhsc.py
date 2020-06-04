import numpy as np
import time
import scipy
import scipy.misc
import scipy.sparse
import matplotlib.pyplot as plt

class DHSC(object):
    """Encapsulates all useful information about a spin half disordered AFM Heisenberg chain
    H = \sum_i S_i \dot S_{i+1} + h_i S_i^z
    """

    def __init__(self, params={}):
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

    def make_hamiltonian(self):
        f = self.h_dis
        self.Hsite = [scipy.sparse.csr_matrix((self.dimH, self.dimH)) for cnt in np.arange(self.N)]
        
        for n in np.arange(self.N):
            h = {}

            if n == 0:
                h["(1, 0)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-f[0])]]))
                h["(1, 1)"] = scipy.sparse.csr_matrix(np.array([[0.5*(f[0])]]))
            else:
                h["(1, 0)"] = scipy.sparse.csr_matrix((1,1))
                h["(1, 1)"] = scipy.sparse.csr_matrix((1,1))
            
            for N1 in np.arange(2, self.N+1):
                for nUp1 in np.arange(np.maximum(0, self.nUp+N1-self.N), np.minimum(self.nUp, N1)+1):
                    bd = np.array([int(scipy.misc.comb(N1-1, nUp1)), #dim of top left block 
                        int(scipy.misc.comb(N1-1, nUp1-1)), #dim of bottom right block
                        int(scipy.misc.comb(N1-2, nUp1)), #number of aligned 00 spins at the surface
                        int(scipy.misc.comb(N1-2, nUp1-1)), #number of antialigned 01/10 spins at the surface
                        int(scipy.misc.comb(N1-2, nUp1-2))]) #number of aligned 11 spins at the surface
                    
                    if n == N1-2: #do some stitching
                        try:
                            BTL = h["({0:d}, {1:d})".format(N1-1, nUp1)] + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[2]), 
                                -0.125*np.ones(bd[3])))], [0], format='csr')
                        except KeyError: BTL = scipy.sparse.csr_matrix((0,0))
                        try:
                            BBR = h["({0:d}, {1:d})".format(N1-1, nUp1-1)] + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[3]),
                                +0.125*np.ones(bd[4])))], [0], format='csr')
                        except KeyError: BBR = scipy.sparse.csr_matrix((0,0))
                        OD = 0.25*scipy.sparse.eye(bd[0], bd[1], -bd[2])

                    elif n == N1-1: #set-up the on-site Hamiltonian
                        try:
                            BTL = h["({0:d}, {1:d})".format(N1-1, nUp1)] + \
                                scipy.sparse.diags([(0.5*(-f[N1-1]))*
                                np.ones(bd[0])], [0], format='csr') + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[2]), 
                                -0.125*np.ones(bd[3])))], [0], format='csr')
                        except KeyError: BTL = scipy.sparse.csr_matrix((0,0))
                        try:
                            BBR = h["({0:d}, {1:d})".format(N1-1, nUp1-1)] + \
                                scipy.sparse.diags([(0.5*(f[N1-1]))*
                                np.ones(bd[1])], [0], format='csr')  + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[3]), 
                                +0.125*np.ones(bd[4])))], [0], format='csr')
                        except KeyError: BBR = scipy.sparse.csr_matrix((0,0))
                        OD = 0.25*scipy.sparse.eye(bd[0], bd[1], -bd[2])

                    else:
                        try:
                            BTL = h["({0:d}, {1:d})".format(N1-1, nUp1)] #top-left
                        except KeyError: BTL = scipy.sparse.csr_matrix((0,0))
                        try:
                            BBR = h["({0:d}, {1:d})".format(N1-1, nUp1-1)] #bottom-right
                        except KeyError: BBR = scipy.sparse.csr_matrix((0,0))
                        OD = scipy.sparse.csr_matrix((bd[0], bd[1]))

                    h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                        [[BTL, OD], [OD.T, BBR]], format='csr')
                    
            self.Hsite[n] = h["({0:d}, {1:d})".format(self.N, self.nUp)]

        self.H = scipy.sparse.csr_matrix((self.dimH, self.dimH))
        for cnt in np.arange(self.N):
            self.H += self.Hsite[cnt]

        if self.matrixType == 'dense':
            self.H = np.array(self.H.todense())

    def diagonalize(self, E=None, k=6):
        if self.matrixType == 'dense':
            self.evals, self.evecs = np.linalg.eigh(self.H)
        if self.matrixType == 'sparse':
            self.evals, self.evecs = scipy.sparse.linalg.eigsh(self.H, k=k, sigma=E)

    def basis_change(self):
        """change the site Hamiltonians from the decimal (Sz) basis to the eigen basis
        Use only in the dense case"""
        self.HsiteUDU = [np.zeros((self.dimH, self.dimH)) for cnt in np.arange(self.N)]
        for cnt in np.arange(self.N):
            self.HsiteUDU[cnt] = np.array(np.dot(self.evecs.T, self.Hsite[cnt].dot(self.evecs)))

    def get_rho0(self):
        """Find the initial density matrix \rho_L(T = 0) tensor \rho_R(T = inf)
        Use only in the dense case"""
        paramsLeft = self.params
        paramsLeft['N'] = self.N/2
        if paramsLeft['N']%2 == 1:
            paramsLeft['nUp'] = 1
        paramsLeft['h_dis'] = self.h_dis[:self.N/2]

        dLeft = DHSC(paramsLeft)
        dLeft.make_hamiltonian()
        dLeft.diagonalize()
        psiL = dLeft.evecs[:,0]

        vLeft = dLeft.vechspace() # decimal states in left Hilbert space (eg. 3,5,6,9,10,12 for N=4)
        vRight = self.vechspace(self.N - dLeft.N, self.nUp-dLeft.nUp) #(eg. 3,5,6,9,10,12)
        vAll = self.vechspace() #(eg. a vector of 70 states: 15, 23, ..., 232, 240)

        vtot_starts = vRight*2**(dLeft.N) + vLeft[0] #gives the starting indices (viz. 51,83,99,147,163,195)
        # left HS has lower significant bits, right HS has higher significant bits

        rho = np.zeros((self.dimH, self.dimH))
        for i, v in enumerate(vtot_starts):
            psiTot = np.zeros(self.dimH)
            ind = np.where(vAll == v)[0][0]
            psiTot[ind:ind+dLeft.dimH] = psiL
            rho += np.outer(psiTot, psiTot)

        self.rhov = rho/len(vRight)

        # now do a basis change to the eigen basis
        self.state0 = (np.dot(self.evecs.T, np.dot(self.rhov, self.evecs)))

    def siteenergy(self, state0, t):
        """find <E_n> = Tr[\rho(t) H_n]
        state0 is a density matrix in the eigen basis, not in the decimal (Sz) basis if dense,
        state0 is a vector in the decimal (Sz) basis if sparse"""
        if self.matrixType == 'dense':
            phases = np.exp(1j*self.evals[:, np.newaxis]*t)
            rho_t = np.einsum('it,ij,jt->ijt', phases.conj(), state0, phases, optimize=True)
            En_t = np.zeros((self.N, len(t)+1), dtype='complex')
            for cnt in np.arange(self.N):
                En_t[cnt, :len(t)] = np.tensordot(rho_t, self.HsiteUDU[cnt], axes=((0,1), (1,0)))
                #np.sum(self.HsiteUDU*rho_t.T) # Tr[AB] = A_ij B_ji
                En_t[cnt, -1] = np.sum(np.diag(state0)*np.diag(self.HsiteUDU[cnt]))    

        if self.matrixType == 'sparse':
            En_t = np.zeros((self.N, len(t)), dtype='complex')
            for i, tim in enumerate(t):
                if tim == 0: vt = state0 
                else:
                    Ht = scipy.sparse.linalg.expm(1j*tim*self.H)
                    vt = Ht.dot(state0)
                for cnt in np.arange(self.N):
                    En_t[cnt,i] = np.dot(vt.conj().T, self.Hsite[cnt].dot(vt))

        return np.real(En_t) # is a self.N x t matrix

    def get_psi0(self):
        """find an initial state that is the ground state on the left and
        in the middle of the spectrum on the right"""
        paramsLeft = self.params
        paramsLeft['N'] = self.N/2
        if paramsLeft['N']%2 == 1:
            paramsLeft['nUp'] = 1
        paramsLeft['h_dis'] = self.h_dis[:self.N/2]

        dLeft = DHSC(paramsLeft)
        dLeft.make_hamiltonian()
        dLeft.diagonalize()
        psiL = dLeft.evecs[:,0]

        paramsRight = self.params
        paramsRight['N'] = self.N - dLeft.N
        paramsRight['nUp'] = self.nUp - dLeft.nUp
        paramsRight['h_dis'] = self.h_dis[self.N/2:]

        dRight = DHSC(paramsRight)
        dRight.make_hamiltonian()
        dRight.diagonalize(E=0, k=1)
        psiR = dRight.evecs[:,0]

        vLeft = dLeft.vechspace() # decimal states in left Hilbert space (eg. 3,5,6,9,10,12 for N=4)
        vRight = dRight.vechspace() #(eg. 3,5,6,9,10,12)
        vAll = self.vechspace() #(eg. a vector of 70 states: 15, 23, ..., 232, 240)

        v0 = np.zeros(self.dimH)

        for i1, n1 in enumerate(vLeft):
            for i2, n2 in enumerate(vRight):
                ind = np.where(vAll == n2*2**(dLeft.N) + n1)[0][0]
                v0[ind] = psiL[i1]*psiR[i2]

        self.state0 = v0

    def vechspace(self, N=None, n=None):
        """Return the decimal representations of all vectors in the relevant Hilbert space"""
        if N == None: N = self.N
        if n == None: n = self.nUp
        if n < 0 or n > N: return np.array([], dtype='int64')
        if N == 1:
            if n == 0: return np.array([0])
            if n == 1: return np.array([1])
        else:
            return np.concatenate((self.vechspace(N-1,n), 2**(N-1)+self.vechspace(N-1, n-1)))

    def dec2spin(self, v):
        """Converts a decimal number to a spin configuration e.g., 13 -> [1, 0, 1, 1, 0]
        (least significant bit on the left)"""
        return np.array([int(x) for x in list(('{0:0'+str(self.N)+'b}').format(v))])[::-1]

    def m_element(self, v1, v2):
        """Return the matrix element < v1 | H | v2 >"""
        spins1 = self.dec2spin(v1)
        spins2 = self.dec2spin(v2)
        if v1 == v2:
            fieldE = 0.5*np.sum(self.h_dis * (2*spins1-1)) # h S^z term
            ndw = np.sum(np.logical_xor(spins1[1:], spins1[:-1])) # number of domain walls
            intE = 0.25*(self.N-1) - 0.5*ndw # S_i^z S_{i+1}^z term
            return fieldE+intE
        flips = np.logical_xor(spins1, spins2)
        if np.sum(flips) == 2 and np.sum(np.logical_and(flips[1:], flips[:-1])) == 1:
            return 0.5
        else:
            return 0

    def E_exp_psi(self, v):
        """Return the energy for any pure state |\psi > 
        the vector v is the list of coefficients in the Sz basis""" 
        return np.dot(v.T, np.dot(self.H,v))

    def r_avg(self, E=None, dE=None):
        """return the Huse-Oganesyan r value of the eigenvalue spectrum,
        E and dE refer to ranges of the raw eigenvalues"""
        evs = (self.evals - self.evals[0])/(self.evals[-1] - self.evals[0])
        if E==None: E_target = evs
        else: E_target = evs[evs > E-dE][evs[evs > E-dE] < E+dE]
        delta = E_target[1:]-E_target[:-1]
        r = np.minimum(delta[1:]/delta[:-1], delta[:-1]/delta[1:])
        return (np.mean(r), len(E_target))

    def get_dsf(self):
        """return the dynamical spin fraction (Pal-Huse eq. (6) / Luitz et al eq. (A3))"""
        if not hasattr(self, 'dsf'):
            if not hasattr(self, 'allvecs'): self.allvecs = self.vechspace()
            v_z = np.zeros((self.dimH, self.N))
            for cnt in np.arange(self.dimH):
                v_z[cnt,:] = (self.dec2spin(self.allvecs[cnt])) - 0.5 # +/- 1/2 for spin up/down
            phase = np.cos(2*np.pi*(np.arange(self.N)+1)/self.N)
            nMn = np.einsum('in, m, im -> n', np.abs(self.evecs)**2, phase, v_z, optimize=True)
            phase2 = phase[:, np.newaxis].conj() * phase
            nMn2 = np.einsum('in, lm, im, il -> n', np.abs(self.evecs)**2, phase2, v_z, v_z, optimize=True)
            self.dsf = np.real(1 - np.abs(nMn)**2/nMn2)

    def get_a1(self):
        """return the participation coefficient a_1 (Luitz Fig. 1): 
        a_1 = -(\sum_i |\psi_i|^2 \ln |\psi_i|^2) / \ln N (= 1 for ergodic, 0 for localized)"""
        if not hasattr(self, 'a1'):
            self.a1 = -np.sum(np.abs(self.evecs)**2 * np.log(np.abs(self.evecs)**2), axis=0) / np.log(self.dimH)

    def get_EE(self):
        """get the bipartite entanglement entropy for the eigenstates"""
        if not hasattr(self, 'EE'):
            n_eig = self.evecs.shape[1]
            if not hasattr(self, 'allvecs'): self.allvecs = self.vechspace()
            self.EE = np.zeros(n_eig)
            for cnt in np.arange(n_eig):
                C = np.zeros(2**(self.N))
                C[self.allvecs] = self.evecs[:,cnt]
                _, s, _ = np.linalg.svd(C.reshape(2**(self.N/2), 2**(self.N/2)))
                self.EE[cnt] = -np.sum(np.abs(s)**2 * np.log(np.abs(s)**2))

if __name__ == "__main__":
    # test only
    tsc = np.array([0,1,2,5,10,100])
    params = {"N": N, "Sz": 0, "W": W, "seed": cnts, "matrixType": 'sparse'}
    d1 = DHSC(params)
    t1 = time.time()
    d1.make_hamiltonian()
    t2 = time.time()
    d1.diagonalize()
    t3 = time.time()
    d1.basis_change()
    t4 = time.time()
    d1.get_rho0()
    t5 = time.time()
    Emin = d1.evals[0]; Emax = d1.evals[-1]
    t = 2*np.pi*tsc/(Emax-Emin)
    t6 = time.time()
    Ent = d1.siteenergy(d1.state0, t)
    t7 = time.time()
    Esc[cnts-1, :,:] = (d1.N*Ent - Emin)/(Emax-Emin)
    print("make H: {0:.3f}, diag: {1:.3f}, basis: {2:.3f}, rho: {3:.3f}, evolve: {4:.3f}".format(t2-t1, t3-t2, t4-t3, t5-t4, t7-t6))