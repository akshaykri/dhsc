import numpy as np
import time
import getopt
import os
import scipy
import matplotlib.pyplot as plt
from dhsc import DHSC


class Simulation(object):
    """Encapsulates everything we need to simulate LLL with disorder / deltas
    """

    def __init__(self, params = {}):
        self.N = params.get("N", 12) # number of sites
        self.W = params.get("W", 8) #disorder strength -W to W
        self.BC = params.get("BC", 'open') #boundary condition
        self.Sz = params.get("Sz", 0) # which total spin sector
        self.matrixType = params.get("matrixType", 'dense')
        self.Niter = params.get("Niter", 10) #number of iterations
        self.tsc = params.get("tsc", np.array([0]))#, 2, 5, 10, 100]))
        self.savdir = params.get("savdir", '~')
        self.params = {"N": self.N, "W": self.W, "BC": self.BC, "Sz": self.Sz, "matrixType": 
            self.matrixType, "Niter": self.Niter}

    def time_dhsc(self):
        if self.matrixType == 'dense': 
            Esc = np.zeros((self.Niter, self.N, len(self.tsc)+1))
        else:
            Esc = np.zeros((self.Niter, self.N, len(self.tsc)))
        for cnts in (np.arange(self.Niter)+1):
            params = self.params
            params['seed'] = cnts
            d1 = DHSC(params)
            t1 = time.time()
            d1.make_hamiltonian()
            t2 = time.time()
            d1.diagonalize()
            t3 = time.time()
            if self.matrixType == 'dense':
                d1.basis_change()
                t4 = time.time()
                d1.get_rho0()
            else:
                d1.get_psi0()
                t5 = time.time()
            Emin = np.min(d1.evals); Emax = np.max(d1.evals)
            t = 2*np.pi*self.tsc/(Emax-Emin)
            t6 = time.time()
            Ent = d1.siteenergy(d1.state0, t)
            t7 = time.time()
            Esc[cnts-1, :,:] = (d1.N*Ent - Emin)/(Emax-Emin)
            if not os.path.isdir(self.savdir+'/N{0:d}'.format(self.N, 10*self.W)):
                os.mkdir(self.savdir+'/N{0:d}'.format(self.N, 10*self.W))
            np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Eall'.format(self.N, 10*self.W))
            #if cnts%20 == 0: print("{0:d} : {1:.3f}".format(cnts, t7-t1))
            if self.matrixType == 'dense':
                print("{5:d} -- make H: {0:.3f}, diag: {1:.3f}, basis: {2:.3f},\
                 rho: {3:.3f}, evolve: {4:.3f}".format(t2-t1, t3-t2, t4-t3, t5-t4, t7-t6, cnts))
            else:
                print("{5:d} -- make H: {0:.3f}, diag: {1:.3f}, \
                 v: {3:.3f}, evolve: {4:.3f}".format(t2-t1, t3-t2, 0, t5-t3, t7-t6, cnts))

        self.Esc_mean = np.mean(Esc, axis=0)
        self.Esc_stderr = np.std(Esc, axis=0)/np.sqrt(self.Niter)
        np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Emean'.format(self.N, 10*self.W))
        np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Estd'.format(self.N, 10*self.W))

    def run_dhsc(self):
        return 0

    def plot_Esc(self):
        plt.figure(1)
        for cnt in np.arange(len(self.Esc_mean[0,:])):
            plt.errorbar(np.arange(self.N)+1, self.Esc_mean[:,cnt], yerr=self.Esc_stderr[:,cnt])
        plt.xlim(0.5, self.N+0.5)
        plt.ylim(-1, 2)
        plt.xlabel(r'Site index')
        plt.ylabel(r'$ \langle \epsilon_n \rangle = \frac{N \langle E_n \rangle - E_{min}}{E_{max}- E_{min}}$')
        labels = []
        for t in self.tsc:
            if t%1 == 0: labels.append(r'$\tilde{t} = '+r'{0:d}$'.format(int(t)))
            else: labels.append(r'$\tilde{t} = '+r'{0:.2f}$'.format(t))
        labels.append(r'$\tilde{t} = \infty$')
        plt.legend(labels=labels, loc='best', fontsize=12)
        plt.title(r'Energy as a function of $\tilde{t} = \frac{t (E_{max} - E_{min})}{2 \pi \hbar}$')
        plt.tight_layout()

def script_args(argv):
    # read in parameters
    par = {}
    try:
        opts, args = getopt.getopt(argv, "", ["savdir=", "N=", "W=", "Niter=", "matrixType="])
    except getopt.GetoptError:
        print('Wrong arguments')
        sys.exit(2)
    for opt, arg in opts:
        if(opt=="--savdir"): par['savdir'] = arg
        if(opt=="--N"): par['N'] = int(arg)
        if(opt=="--W"): par['W'] = float(arg)
        if(opt=="--Niter"): par['Niter'] = int(arg)
        if(opt=="--matrixType"): par['matrixType'] = arg
    
    s1 = Simulation(par)
    s1.time_dhsc()
    s1.plot_Esc()

if __name__ == "__main__":
    script_args(sys.argv[1:])
    