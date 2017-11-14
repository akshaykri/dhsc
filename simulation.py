import numpy as np
import time
import getopt
import os
import sys
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
        self.tsc = params.get("tsc", np.array([0, 1, 2, 5, 10, 100]))
        self.savdir = params.get("savdir", '/mnt/cmcomp/ak20/spinchain')
        self.params = {"N": self.N, "W": self.W, "BC": self.BC, "Sz": self.Sz, "matrixType": 
            self.matrixType, "Niter": self.Niter}

    def time_dhsc(self):
        if self.matrixType == 'dense': 
            Esc = np.zeros((self.Niter, self.N, len(self.tsc)+1))
        else:
            Esc = np.zeros((self.Niter, self.N, len(self.tsc)))

        lastseed = 0
        Escfile = self.savdir+'/N{0:d}/W{1:02.0f}_Eall.npy'.format(self.N, 10*self.W)
        if os.path.isfile(Escfile):
            Esc = np.load(Escfile)
            lastseed = np.where(Esc[:,0,0] == 0)[0][0]

        for cnts in (np.arange(lastseed, self.Niter)+1):
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
            np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Eall'.format(self.N, 10*self.W), Esc)
            #if cnts%20 == 0: print("{0:d} : {1:.3f}".format(cnts, t7-t1))
            if self.matrixType == 'dense':
                print("{5:d} -- make H: {0:.3f}, diag: {1:.3f}, basis: {2:.3f},\
                 rho: {3:.3f}, evolve: {4:.3f}".format(t2-t1, t3-t2, t4-t3, t5-t4, t7-t6, cnts))
            else:
                print("{5:d} -- make H: {0:.3f}, diag: {1:.3f}, \
                 v: {3:.3f}, evolve: {4:.3f}".format(t2-t1, t3-t2, 0, t5-t3, t7-t6, cnts))

        self.Esc_mean = np.mean(Esc, axis=0)
        self.Esc_stderr = np.std(Esc, axis=0)/np.sqrt(self.Niter)
        np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Emean'.format(self.N, 10*self.W), self.Esc_mean)
        np.save(self.savdir+'/N{0:d}/W{1:02.0f}_Estd'.format(self.N, 10*self.W), self.Esc_stderr)

    def run_dhsc(self):
        return 0

    def plot_Esc(self, ts=np.array([-1]), avg=True):
        if ts.all() == -1: ts = self.tsc
        self.Esc_mean = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Emean.npy'.format(self.N, 10*self.W))
        self.Esc_stderr = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Estd.npy'.format(self.N, 10*self.W))
        self.Escseed1 = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Eall.npy'.format(self.N, 10*self.W))[0,:,:]
        plt.figure(1)
        for cnt in np.arange(len(self.tsc)):
            if (self.tsc[cnt] == ts).any() or cnt == len(self.tsc)-1:
                if avg == True:
                    plt.errorbar(np.arange(self.N)+1, self.Esc_mean[:,cnt], yerr=self.Esc_stderr[:,cnt], marker='o')
                else:
                    plt.plot(np.arange(self.N)+1, self.Escseed1[:,cnt], marker='o')
        plt.xlim(0.5, self.N+0.5)
        if avg == True: plt.ylim(-0.1, 0.6)
        plt.xlabel(r'Site index')
        plt.ylabel(r'$ \langle \epsilon_n \rangle = \frac{N \langle E_n \rangle - E_{min}}{E_{max}- E_{min}}$')
        labels = []
        for t in ts:
            if t%1 == 0: labels.append(r'$\tilde{t} = '+r'{0:d}$'.format(int(t)))
            else: labels.append(r'$\tilde{t} = '+r'{0:.2f}$'.format(t))
        labels.append(r'$\tilde{t} = \infty$')
        plt.legend(labels=labels, loc='best', fontsize=12)
        plt.title(r'Energy as a function of $\tilde{t} = \frac{t (E_{max} - E_{min})}{2 \pi \hbar}$')
        plt.tight_layout()

    def plot_Trelax(self):
        W_arr = np.arange(16)*0.5 + 0.5
        y = np.zeros(len(W_arr))*np.nan
        dy = np.zeros(len(W_arr))*np.nan
        for cnt, W in enumerate(W_arr):
            if os.path.isfile(self.savdir+'/N{0:d}/W{1:02.0f}_Emean.npy'.format(self.N, 10*W)):
                Em = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Emean.npy'.format(self.N, 10*W))
                Es = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Estd.npy'.format(self.N, 10*W))
                print('W = {0:.1f}'.format(W))
                print('T_R (f) = {0:.4f} +/- {1:.4f}; T_L (f) = {2:.4f} +/- {3:.4f}'.format(Em[-2, -1], Es[-2, -1], Em[1, -1], Es[1, -1]))
                print('T_R (i) = {0:.4f} +/- {1:.4f}; T_L (i) = {2:.4f} +/- {3:.4f}'.format(Em[-2, 0], Es[-2, 0], Em[1, 0], Es[1, 0]))
                y[cnt] = (Em[-2, -1] - Em[1, -1])/((Em[-2, 0] - Em[1, 0]))
                dy[cnt] = ((Es[-2, -1] + Es[1, -1])/(Em[-2, -1] - Em[1, -1]) +
                     (Es[-2, 0] + Es[1, 0])/(Em[-2, 0] - Em[1, 0])) * y[cnt]

        plt.figure(1)
        plt.errorbar(W_arr, y, yerr=dy, marker='o')
        plt.ylim(0, 1.1)
        plt.xlabel(r'Disorder $W$')
        plt.ylabel(r'$ \frac{\langle \epsilon_{R}(t = \infty) - \epsilon_{L}(t = \infty) \rangle}{\langle \epsilon_{R}(t = 0) - \epsilon_{L}(t = 0) \rangle}  $')
        plt.title(r'Energy relaxation as a function of disorder')
        plt.tight_layout()

    def plot_Trelax2(self):
        W_arr = np.arange(16)*0.5 + 0.5
        y = np.zeros(len(W_arr))*np.nan
        dy = np.zeros(len(W_arr))*np.nan
        for cnt, W in enumerate(W_arr):
            if os.path.isfile(self.savdir+'/N{0:d}/W{1:02.0f}_Emean.npy'.format(self.N, 10*W)):
                Eall = np.load(self.savdir+'/N{0:d}/W{1:02.0f}_Eall.npy'.format(self.N, 10*W))
                ELi = Eall[:,1,0]; ELf = Eall[:,1,-1]; ERi = Eall[:,-2,0]; ERf = Eall[:,-2,-1]
                q = np.vstack((ELi, ELf, ERi, ERf)); qm = np.mean(q, axis=1)
                qs = np.cov(q)
                print('W = {0:.1f}'.format(W))
                print('T_R (f) = {0:.4f} +/- {1:.4f}; T_L (f) = {2:.4f} +/- {3:.4f}'.format(qm[3], 
                    np.sqrt(qs[3,3]/self.Niter), qm[1], np.sqrt(qs[1,1]/self.Niter)))
                print('T_R (i) = {0:.4f} +/- {1:.4f}; T_L (i) = {2:.4f} +/- {3:.4f}'.format(qm[2], 
                    np.sqrt(qs[2,2]/self.Niter), qm[0], np.sqrt(qs[0,0]/self.Niter)))
                y[cnt] = (qm[3]-qm[1])/(qm[2]-qm[0])
                vcoeff = np.array([1/(qm[2]-qm[0]), -1/(qm[2]-qm[0]), 
                    -(qm[3]-qm[1])/(qm[2]-qm[0])**2, (qm[3]-qm[1])/(qm[2]-qm[0])**2])
                dy[cnt] = np.sqrt(np.dot(vcoeff, np.dot(qs, vcoeff))/self.Niter)

        plt.figure(1)
        plt.errorbar(W_arr, y, yerr=dy, marker='o')
        plt.ylim(0, 1.1)
        plt.xlabel(r'Disorder $W$')
        plt.ylabel(r'$ \frac{\langle \epsilon_{R}(t = \infty) - \epsilon_{L}(t = \infty) \rangle}{\langle \epsilon_{R}(t = 0) - \epsilon_{L}(t = 0) \rangle}  $')
        plt.title(r'Energy relaxation as a function of disorder')
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
    #s1.plot_Esc()

if __name__ == "__main__":
    script_args(sys.argv[1:])
    