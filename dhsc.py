import numpy as np
import time
import scipy

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

    def make_hamiltonian(self):
        # dynamic programming
        h = {}
        if self.matrixType == 'dense':
            if self.N%2 == 0:
                h["(2, 0)"] = np.array([[0.5*(-self.h_dis[0]-self.h_dis[1])+0.25]])
                h["(2, 1)"] = np.array([[0.5*(self.h_dis[0]-self.h_dis[1])-0.25, 0.5], 
                    [0.5, 0.5*(-self.h_dis[0]+self.h_dis[1])-0.25]])
                h["(2, 2)"] = np.array([[0.5*(self.h_dis[0]+self.h_dis[1])+0.25]])
            else:
                h["(1, 0)"] = np.array([[0.5*(-self.h_dis[0])]])
                h["(1, 1)"] = np.array([[0.5*(+self.h_dis[0])]])

            for N1 in np.arange(4 - self.N%2, self.N+2, 2):
                for nUp1 in np.arange(np.maximum(0, self.nUp+N1-self.N), np.minimum(self.nUp, N1)+1):
                    bd = np.array([int(scipy.misc.comb(N1-2, nUp1)), #dim of top left block 
                        int(scipy.misc.comb(N1-2, nUp1-1)), #dim of middle two blocks
                        int(scipy.misc.comb(N1-2, nUp1-2)), #dim of bottom right blocks
                        int(scipy.misc.comb(N1-3, nUp1)), #diag offset for OD_0001
                        int(scipy.misc.comb(N1-3, nUp1-1)), #diag offset for OD_1011
                        int(scipy.misc.comb(N1-3, nUp1-2)), #aligned terms in B01
                        int(scipy.misc.comb(N1-3, nUp1-3))]) #aligned terms in B11
                    try:
                        B00 = h["({0:d}, {1:d})".format(N1-2, nUp1)] + \
                            np.diag((0.5*(-self.h_dis[N1-2]-self.h_dis[N1-1])+0.25)*np.ones(bd[3]+bd[4])) + \
                            np.diag(np.concatenate((+0.25*np.ones(bd[3]), -0.25*np.ones(bd[4]))))
                    except KeyError: B00 = np.random.rand(0,0)
                    try:
                        B01 = h["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                            np.diag((0.5*(self.h_dis[N1-2]-self.h_dis[N1-1])-0.25)*np.ones(bd[4]+bd[5]))+ \
                            np.diag(np.concatenate((-0.25*np.ones(bd[4]), +0.25*np.ones(bd[5]))))
                    except KeyError: B01 = np.random.rand(0,0)
                    try:
                        B10 = h["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                            np.diag((0.5*(-self.h_dis[N1-2]+self.h_dis[N1-1])-0.25)*np.ones(bd[4]+bd[5]))+ \
                            np.diag(np.concatenate((+0.25*np.ones(bd[4]), -0.25*np.ones(bd[5]))))
                    except KeyError: B10 = np.random.rand(0,0)
                    try:
                        B11 = h["({0:d}, {1:d})".format(N1-2, nUp1-2)] + \
                            np.diag((0.5*(self.h_dis[N1-2]+self.h_dis[N1-1])+0.25)*np.ones(bd[5]+bd[6]))+ \
                            np.diag(np.concatenate((-0.25*np.ones(bd[5]), +0.25*np.ones(bd[6]))))
                    except KeyError: B11 = np.random.rand(0,0)
                    OD_0001 = 0.5*np.eye(bd[0], bd[1], -bd[3])
                    OD_0110 = 0.5*np.eye(bd[1], bd[1])
                    OD_1011 = 0.5*np.eye(bd[1], bd[2], -bd[4])
                    h["({0:d}, {1:d})".format(N1, nUp1)] = np.block(
                        [[B00, OD_0001, np.zeros((bd[0], bd[1]+bd[2]))],
                         [OD_0001.T, B01, OD_0110, np.zeros((bd[1], bd[2]))],
                         [np.zeros((bd[1], bd[0])), OD_0110, B10, OD_1011],
                         [np.zeros((bd[2], bd[0]+bd[1])), OD_1011.T, B11]])

        elif self.matrixType == 'sparse':
            if self.N%2 == 0:
                h["(2, 0)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-self.h_dis[0]-self.h_dis[1])+0.25]]))
                h["(2, 1)"] = scipy.sparse.csr_matrix(np.array([[0.5*(self.h_dis[0]-self.h_dis[1])-0.25, 0.5], 
                    [0.5, 0.5*(-self.h_dis[0]+self.h_dis[1])-0.25]]))
                h["(2, 2)"] = scipy.sparse.csr_matrix(np.array([[0.5*(self.h_dis[0]+self.h_dis[1])+0.25]]))
            else:
                h["(1, 0)"] = scipy.sparse.csr_matrix(np.array([[-self.h_dis[0]]]))
                h["(1, 1)"] = scipy.sparse.csr_matrix(np.array([[+self.h_dis[0]]]))

            for N1 in np.arange(4 - self.N%2, self.N+2, 2):
                for nUp1 in np.arange(np.maximum(0, self.nUp+N1-self.N), np.minimum(self.nUp, N1)+1):
                    bd = np.array([int(scipy.misc.comb(N1-2, nUp1)), #dim of top left block 
                        int(scipy.misc.comb(N1-2, nUp1-1)), #dim of middle two blocks
                        int(scipy.misc.comb(N1-2, nUp1-2)), #dim of bottom right block
                        int(scipy.misc.comb(N1-3, nUp1)), #diag offset for OD_0001
                        int(scipy.misc.comb(N1-3, nUp1-1)), #diag offset for OD_1011
                        int(scipy.misc.comb(N1-3, nUp1-2)), #aligned terms in B01
                        int(scipy.misc.comb(N1-3, nUp1-3))]) #aligned terms in B11
                    try:
                        B00 = h["({0:d}, {1:d})".format(N1-2, nUp1)] + \
                            scipy.sparse.diags([(0.5*(-self.h_dis[N1-2]-self.h_dis[N1-1])+0.25)*
                                np.ones(bd[3]+bd[4])], [0], format='csr') + \
                            scipy.sparse.diags([np.concatenate((+0.25*np.ones(bd[3]), 
                                -0.25*np.ones(bd[4])))], [0], format='csr')
                    except KeyError: B00 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                    try:
                        B01 = h["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                            scipy.sparse.diags([(0.5*(self.h_dis[N1-2]-self.h_dis[N1-1])-0.25)*
                                np.ones(bd[4]+bd[5])], [0], format='csr') + \
                            scipy.sparse.diags([np.concatenate((-0.25*np.ones(bd[4]),
                                +0.25*np.ones(bd[5])))], [0], format='csr')
                    except KeyError: B01 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                    try:
                        B10 = h["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                            scipy.sparse.diags([(0.5*(-self.h_dis[N1-2]+self.h_dis[N1-1])-0.25)*
                                np.ones(bd[4]+bd[5])], [0], format='csr') + \
                            scipy.sparse.diags([np.concatenate((+0.25*np.ones(bd[4]),
                                -0.25*np.ones(bd[5])))], [0], format='csr')
                    except KeyError: B10 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                    try:
                        B11 = h["({0:d}, {1:d})".format(N1-2, nUp1-2)] + \
                            scipy.sparse.diags([(0.5*(self.h_dis[N1-2]+self.h_dis[N1-1])+0.25)*
                                np.ones(bd[5]+bd[6])], [0], format='csr') + \
                            scipy.sparse.diags([np.concatenate((-0.25*np.ones(bd[5]),
                                +0.25*np.ones(bd[6])))], [0], format='csr')
                    except KeyError: B11 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                    OD_0001 = 0.5*scipy.sparse.eye(bd[0], bd[1], -bd[3])
                    OD_0110 = 0.5*scipy.sparse.eye(bd[1], bd[1])
                    OD_1011 = 0.5*scipy.sparse.eye(bd[1], bd[2], -bd[4])
                    if bd[0] != 0:
                        if bd[1] != 0:
                            if bd[2] != 0:
                                h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                    [[B00, OD_0001, None, None],
                                     [OD_0001.T, B01, OD_0110, None],
                                     [None, OD_0110, B10, OD_1011],
                                     [None, None, OD_1011.T, B11]])
                            else:
                                h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                    [[B00, OD_0001, None],
                                     [OD_0001.T, B01, OD_0110],
                                     [None, OD_0110, B10]])
                        else:
                            h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B00]])
                    else:
                        if bd[1] != 0:
                            h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B01, OD_0110, None],
                                 [OD_0110, B10, OD_1011],
                                 [None, OD_1011.T, B11]])
                        else:
                            h["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B11]])

        self.H = h["({0:d}, {1:d})".format(self.N, self.nUp)] 

    def make_hamiltonian_2(self):
        self.decvecs = self.vechspace()
        f = self.h_dis
        self.H2 = [scipy.sparse.coo_matrix((self.dimH, self.dimH)) for cnt in np.arange(self.N)]
        
        h = [{} for cnt in np.arange(self.N)]

        for n in np.arange(self.N):
            if self.N%2 == 0:
                if n == 0:
                    h[n]["(2, 0)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-f[0])+0.125]]))
                    h[n]["(2, 1)"] = scipy.sparse.csr_matrix(np.array([[0.5*(f[0])-0.125, 0.25], 
                        [0.25, 0.5*(-f[0])-0.125]]))
                    h[n]["(2, 2)"] = scipy.sparse.csr_matrix(np.array([[0.5*(f[0])+0.125]]))
                elif n == 1:
                    h[n]["(2, 0)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-f[1])+0.125]]))
                    h[n]["(2, 1)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-f[1])-0.125, 0.25], 
                        [0.25, 0.5*(f[1])-0.125]]))
                    h[n]["(2, 2)"] = scipy.sparse.csr_matrix(np.array([[0.5*(f[1])+0.125]]))
                else:
                    h[n]["(2, 0)"] = scipy.sparse.csr_matrix((1,1))
                    h[n]["(2, 1)"] = scipy.sparse.csr_matrix((2,2))
                    h[n]["(2, 2)"] = scipy.sparse.csr_matrix((1,1))
        
            else:
                if n == 0:
                    h[n]["(1, 0)"] = scipy.sparse.csr_matrix(np.array([[0.5*(-f[0])]]))
                    h[n]["(1, 1)"] = scipy.sparse.csr_matrix(np.array([[0.5*(f[0])]]))
                else:
                    h[n]["(1, 0)"] = scipy.sparse.csr_matrix((1,1))
                    h[n]["(1, 1)"] = scipy.sparse.csr_matrix((1,1))

            for N1 in np.arange(4 - self.N%2, self.N+2, 2):
                for nUp1 in np.arange(np.maximum(0, self.nUp+N1-self.N), np.minimum(self.nUp, N1)+1):
                    bd = np.array([int(scipy.misc.comb(N1-2, nUp1)), #dim of top left block 
                        int(scipy.misc.comb(N1-2, nUp1-1)), #dim of middle two blocks
                        int(scipy.misc.comb(N1-2, nUp1-2)), #dim of bottom right block
                        int(scipy.misc.comb(N1-3, nUp1)), #diag offset for OD_0001
                        int(scipy.misc.comb(N1-3, nUp1-1)), #diag offset for OD_1011
                        int(scipy.misc.comb(N1-3, nUp1-2)), #aligned terms in B01
                        int(scipy.misc.comb(N1-3, nUp1-3))]) #aligned terms in B11
                    if n < N1-3 or n > N1-1:     
                        try:
                            B00 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1)]
                        except KeyError: B00 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B01 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)]
                        except KeyError: B01 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B10 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)]
                        except KeyError: B10 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B11 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-2)]
                        except KeyError: B11 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        OD_0001 = scipy.sparse.csr_matrix((bd[0], bd[1]))
                        OD_0110 = scipy.sparse.csr_matrix((bd[1], bd[1]))
                        OD_1011 = scipy.sparse.csr_matrix((bd[1], bd[2]));

                    elif n == N1-3:
                        try:
                            B00 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1)] + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[3]), 
                                    -0.125*np.ones(bd[4])))], [0], format='csr')
                        except KeyError: B00 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B01 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[4]),
                                    +0.125*np.ones(bd[5])))], [0], format='csr')
                        except KeyError: B01 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B10 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[4]),
                                    -0.125*np.ones(bd[5])))], [0], format='csr')
                        except KeyError: B10 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B11 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-2)] + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[5]),
                                    +0.125*np.ones(bd[6])))], [0], format='csr')
                        except KeyError: B11 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        OD_0001 = 0.25*scipy.sparse.eye(bd[0], bd[1], -bd[3])
                        OD_0110 = scipy.sparse.csr_matrix((bd[1], bd[1]))
                        OD_1011 = 0.25*scipy.sparse.eye(bd[1], bd[2], -bd[4])

                    elif n == N1-2:
                        try:
                            B00 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1)] + \
                                scipy.sparse.diags([(0.5*(-f[N1-2])+0.125)*
                                    np.ones(bd[3]+bd[4])], [0], format='csr') + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[3]), 
                                    -0.125*np.ones(bd[4])))], [0], format='csr')
                        except KeyError: B00 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B01 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([(0.5*(f[N1-2])-0.125)*
                                    np.ones(bd[4]+bd[5])], [0], format='csr') + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[4]),
                                    +0.125*np.ones(bd[5])))], [0], format='csr')
                        except KeyError: B01 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B10 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([(0.5*(-f[N1-2])-0.125)*
                                    np.ones(bd[4]+bd[5])], [0], format='csr') + \
                                scipy.sparse.diags([np.concatenate((+0.125*np.ones(bd[4]),
                                    -0.125*np.ones(bd[5])))], [0], format='csr')
                        except KeyError: B10 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B11 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-2)] + \
                                scipy.sparse.diags([(0.5*(f[N1-2])+0.125)*
                                    np.ones(bd[5]+bd[6])], [0], format='csr') + \
                                scipy.sparse.diags([np.concatenate((-0.125*np.ones(bd[5]),
                                    +0.125*np.ones(bd[6])))], [0], format='csr')
                        except KeyError: B11 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        OD_0001 = 0.25*scipy.sparse.eye(bd[0], bd[1], -bd[3])
                        OD_0110 = 0.25*scipy.sparse.eye(bd[1], bd[1])
                        OD_1011 = 0.25*scipy.sparse.eye(bd[1], bd[2], -bd[4])
                    elif n == N1-1:
                        try:
                            B00 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1)] + \
                                scipy.sparse.diags([(0.5*(-f[N1-1])+0.125)*
                                    np.ones(bd[3]+bd[4])], [0], format='csr')
                        except KeyError: B00 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B01 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([(0.5*(-f[N1-1])-0.125)*
                                    np.ones(bd[4]+bd[5])], [0], format='csr')
                        except KeyError: B01 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B10 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-1)] + \
                                scipy.sparse.diags([(0.5*(f[N1-1])-0.125)*
                                    np.ones(bd[4]+bd[5])], [0], format='csr')
                        except KeyError: B10 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        try:
                            B11 = h[n]["({0:d}, {1:d})".format(N1-2, nUp1-2)] + \
                                scipy.sparse.diags([(0.5*(f[N1-1])+0.125)*
                                    np.ones(bd[5]+bd[6])], [0], format='csr')
                        except KeyError: B11 = scipy.sparse.csr_matrix(np.random.rand(0,0))
                        OD_0001 = scipy.sparse.csr_matrix((bd[0], bd[1]))
                        OD_0110 = 0.25*scipy.sparse.eye(bd[1], bd[1])
                        OD_1011 = scipy.sparse.csr_matrix((bd[1], bd[2]))

                    if bd[0] != 0:
                        if bd[1] != 0:
                            if bd[2] != 0:
                                h[n]["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                    [[B00, OD_0001, None, None],
                                     [OD_0001.T, B01, OD_0110, None],
                                     [None, OD_0110, B10, OD_1011],
                                     [None, None, OD_1011.T, B11]])
                            else:
                                h[n]["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                    [[B00, OD_0001, None],
                                     [OD_0001.T, B01, OD_0110],
                                     [None, OD_0110, B10]])
                        else:
                            h[n]["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B00]])
                    else:
                        if bd[1] != 0:
                            h[n]["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B01, OD_0110, None],
                                 [OD_0110, B10, OD_1011],
                                 [None, OD_1011.T, B11]])
                        else:
                            h[n]["({0:d}, {1:d})".format(N1, nUp1)] = scipy.sparse.bmat(
                                [[B11]])

            self.H2[n] = h[n]["({0:d}, {1:d})".format(self.N, self.nUp)]

        self.H2_sum = scipy.sparse.coo_matrix((self.dimH, self.dimH))
        for cnt in np.arange(self.N):
            self.H2_sum += self.H2[cnt]


    def diagonalize(self):
        if self.matrixType == 'dense':
            evals, evecs = np.linalg.eigh(self.H)
            self.evals = evals
            self.evecs = evecs
        if self.matrixType == 'sparse':
            evals, evecs = scipy.sparse.linalg.eigh(self.H)

    def vechspace(self, N=None, n=None):
        """Return the decimal representations of all vectors in the relevant Hilbert space"""
        if N == None: N = self.N
        if n == None: n = self.nUp
        if n < 0 or n > N: return np.array([], dtype='int64')
        if N == 2:
            if n == 0: return np.array([0])
            if n == 1: return np.array([1,2])
            if n == 2: return np.array([3])
        else: return np.concatenate((self.vechspace(N-2, n), 2**(N-2)+self.vechspace(N-2,n-1), 
            2**(N-1)+self.vechspace(N-2, n-1), (3*2**(N-2))+self.vechspace(N-2,n-2)))

    def dec2spin(self, v):
        """Converts a decimal number to a spin configuration e.g., 13 -> [1, 0, 1, 1, 0]
        (least significant bit on the left)"""
        return np.array([int(x) for x in list(('{0:0'+str(self.N)+'b}').format(v))])[::-1]

    def m_element(self, v1, v2):
        """Return the matrix element < v1 | H | v2 >"""
        spins1 = self.dec2spin(v1)
        spins2 = self.dec2spin(v2)
        if v1 == v2:
            fieldE = np.sum(self.h_dis * (2*spins1-1)) # h S^z term
            ndw = np.sum(np.logical_xor(spins1[1:], spins1[:-1])) # number of domain walls
            intE = 0.25*(self.N-1) - 0.5*ndw # S_i^z S_{i+1}^z term
            return fieldE+intE
        flips = np.logical_xor(spins1, spins2)
        if np.sum(flips) == 2 and np.sum(np.logical_and(flips[1:], flips[:-1])) == 1:
            return 0.5
        else:
            return 0

    def E_exp_psi(self, v):
        """Return the site-resolved energy density for any pure state |\psi > 
        (vectorized and site-resolved version of m_element)"""
        # the vector v is the list of coefficients in the Sz basis

    def r_avg(self, E=0.5, dE=0.05):
        evs = (self.evals - self.evals[0])/(self.evals[-1] - self.evals[0])
        E_target = evs[evs > E-dE][evs[evs > E-dE] < E+dE]
        delta = E_target[1:]-E_target[:-1]
        r = np.minimum(delta[1:]/delta[:-1], delta[:-1]/delta[1:])
        return np.mean(r)

if __name__ == "__main__":
    d1 = DHSC()
    d1.make_hamiltonian()

