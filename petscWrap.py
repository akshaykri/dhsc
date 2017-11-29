import sys, time, getopt
import numpy as np
smart = False
if smart == True:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
import scipy.sparse
import petsc, petsc4py, slepc, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

import dhsc

def doSlepc(params):
    paramsdhsc = {'N': 16, 'W': 3.0, 'matrixType': 'sparse'}
    t2 = time.time()
    sH = doPetsc(paramsdhsc, params)
    t3 = time.time()

    E = SLEPc.EPS();
    E.create()
    E.setOperators(sH)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    #E.setFromOptions() 

    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #default eigensolver
    if params.has_key('eps_type'):
        if params['eps_type'] == 'jd' : E.setType(SLEPc.EPS.Type.JD)
        if params['eps_type'] == 'gd' : E.setType(SLEPc.EPS.Type.GD)
    E.setDimensions(6, PETSc.DECIDE) # default get 6 eigenvalues
    if params.has_key('eps_nev'):
        E.setDimensions(params['eps_nev'], PETSc.DECIDE)
    if params.has_key('eps_target'):
        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
        E.setTarget(params['eps_target'])
    if params.has_key('st_type'):
        if params['st_type'] == 'sinvert': E.getST().setType(SLEPc.ST.Type.SINVERT)
        if params['st_type'] == 'cayley': E.getST().setType(SLEPc.ST.Type.CAYLEY)
    if params.has_key('st_ksp_type'):
        if params['st_ksp_type'] == 'gmres': E.getST().getKSP().setType(PETSc.KSP.Type.GMRES)
    if params.has_key('st_pc_type'):
        if params['st_pc_type'] == 'lu': E.getST().getKSP().getPC().setType(PETSc.PC.Type.LU)
        if params['st_pc_type'] == 'jacobi': E.getST().getKSP().getPC().setType(PETSc.PC.Type.JACOBI)
    if params.has_key('st_ksp_max_it'):
        E.getST().getKSP().setTolerances(max_it=params['st_ksp_max_it'])
    if params.has_key('st_pc_factor_mat_solver_package'):
        E.getST().getKSP().getPC().setFactorSolverPackage(params['st_pc_factor_mat_solver_package'])

    E.solve()
    t4 = time.time()

    #copied verbatim from the example file

    Print = PETSc.Sys.Print
    if params.has_key('eps_view'):
        if params['eps_view'] == True: E.view()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    if nconv > 0:
        # Create the results vectors
        vr, wr = sH.getVecs()
        vi, wi = sH.getVecs()
        #
        Print()
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)
            if k.imag != 0.0:
                Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f      %12g" % (k.real, error))

    print("PETSc stuff      : {0:.3f}s\n".format(t3-t2)+
          "SLEPc solution   : {0:.3f}s".format(t4-t3))

def doPetsc(paramsdhsc, params):
    d1 = dhsc.DHSC(paramsdhsc)
    t1 = time.time()
    d1.make_hamiltonian()
    t2 = time.time()

    if smart == True:
        sH = PETSc.Mat().create(comm=comm)
        sH.setSizes([d1.dimH, d1.dimH])
        sH.setUp()
        rstart, rend = sH.getOwnershipRange()
        print("rstart = {0:d}, rend = {1:d}".format(rstart, rend))
        sH = PETSc.Mat().createAIJ(size=d1.H.shape, csr=(d1.H.indptr[rstart:rend+1] - d1.H.indptr[rstart],
            d1.H.indices[d1.H.indptr[rstart]:d1.H.indptr[rend]],
            d1.H.data[d1.H.indptr[rstart]:d1.H.indptr[rend]]), comm=comm)
        
        t3 = time.time()

    else:#be stupid
        sH = PETSc.Mat().createAIJ(size=d1.H.shape, csr=(d1.H.indptr, d1.H.indices, d1.H.data))
        t3 = time.time()

    d1.diagonalize(E = params['eps_target'])
    t4 = time.time()

    print("scipy assembly   : {0:.6f}s\n".format(t2-t1)+
          "scipy to petsc   : {0:.6f}s\n".format(t3-t2)+
          "scipy diagonalize: {0:.6f}s\n".format(t4-t3))

    return sH

def scipyToPetsc(d1, params):
    #params contains SLEPc-specific parameters
    if smart == True:
        sH = PETSc.Mat().create(comm=comm)
        sH.setSizes([d1.dimH, d1.dimH])
        sH.setUp()
        rstart, rend = sH.getOwnershipRange()
        print("rstart = {0:d}, rend = {1:d}".format(rstart, rend))
        sH = PETSc.Mat().createAIJ(size=d1.H.shape, csr=(d1.H.indptr[rstart:rend+1] - d1.H.indptr[rstart],
            d1.H.indices[d1.H.indptr[rstart]:d1.H.indptr[rend]],
            d1.H.data[d1.H.indptr[rstart]:d1.H.indptr[rend]]), comm=comm)

    else:#be stupid and don't use MPI
        sH = PETSc.Mat().createAIJ(size=d1.H.shape, csr=(d1.H.indptr, d1.H.indices, d1.H.data))

    E = SLEPc.EPS();
    E.create()
    E.setOperators(sH)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    #E.setFromOptions() 

    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #default eigensolver
    if params.has_key('eps_type'):
        if params['eps_type'] == 'jd' : E.setType(SLEPc.EPS.Type.JD)
        if params['eps_type'] == 'gd' : E.setType(SLEPc.EPS.Type.GD)
    E.setDimensions(6, PETSc.DECIDE) # default get 6 eigenvalues
    if params.has_key('eps_nev'):
        E.setDimensions(params['eps_nev'], PETSc.DECIDE)
    E.setWhichEigenpairs(E.Which.SMALLEST_MAGNITUDE) #default : find smallest eigenvalues
    if params.has_key('eps_target'):
        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
        E.setTarget(params['eps_target'])
    E.getST().setType(SLEPc.ST.Type.SINVERT) #default mode = shift-invert
    if params.has_key('st_type'):
        if params['st_type'] == 'sinvert': E.getST().setType(SLEPc.ST.Type.SINVERT)
        if params['st_type'] == 'cayley': E.getST().setType(SLEPc.ST.Type.CAYLEY)
    if params.has_key('st_ksp_type'):
        if params['st_ksp_type'] == 'gmres': E.getST().getKSP().setType(PETSc.KSP.Type.GMRES)
    if params.has_key('st_pc_type'):
        if params['st_pc_type'] == 'lu': E.getST().getKSP().getPC().setType(PETSc.PC.Type.LU)
        if params['st_pc_type'] == 'jacobi': E.getST().getKSP().getPC().setType(PETSc.PC.Type.JACOBI)
    if params.has_key('st_ksp_max_it'):
        E.getST().getKSP().setTolerances(max_it=params['st_ksp_max_it'])
    if params.has_key('st_pc_factor_mat_solver_package'):
        E.getST().getKSP().getPC().setFactorSolverPackage(params['st_pc_factor_mat_solver_package'])



def script_args(argv):
    # read in parameters
    par = {}
    try:
        opts, args = getopt.getopt(argv, "", ["eps_nev=", "eps_target=", "eps_type=", "st_type=",
            "st_ksp_type=", "st_pc_type=", "st_ksp_max_it=", "st_pc_factor_mat_solver_package=", "eps_view"])
    except getopt.GetoptError:
        print('Wrong arguments')
        sys.exit(2)
    for opt, arg in opts:
        if(opt=="--eps_nev"): par['eps_nev'] = int(arg)
        if(opt=="--eps_target"): par['eps_target'] = float(arg)
        if(opt=="--eps_type"): par['eps_type'] = arg
        if(opt=="--st_type"): par['st_type'] = arg
        if(opt=="--st_ksp_type"): par['st_ksp_type'] = arg
        if(opt=="--st_pc_type"): par['st_pc_type'] = arg
        if(opt=="--st_ksp_max_it"): par['st_ksp_max_it'] = int(arg)
        if(opt=="--st_pc_factor_mat_solver_package"): par['st_pc_factor_mat_solver_package'] = arg
        if(opt=="--eps_view"): par['eps_view'] = True
        
    doSlepc(par)

if __name__ == '__main__':
    script_args(sys.argv[1:])