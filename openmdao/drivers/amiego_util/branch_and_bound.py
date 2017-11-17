"""
Class definition for the Branch_and_Bound subdriver. This driver can only be run
plugged into the AMIEGO driver.

This is the branch and bound algorithm that maximizes the constrained
expected improvement function and returns an integer infill point. The
algorithm uses the relaxation techniques proposed by Jones et.al. on their
paper on EGO,1998. This enables the algorithm to use any gradient-based
approach to obtain a global solution. Also, to satisfy the integer
constraints, a new branching scheme has been implemented.

Developed by Satadru Roy
School of Aeronautics & Astronautics
Purdue University, West Lafayette, IN 47906
July, 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""

from __future__ import print_function

from collections import OrderedDict
from time import time
from six import iteritems
from six.moves import range

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
from pyDOE import lhs

from openmdao.core.driver import Driver
from openmdao.drivers.amiego_util.genetic_algorithm import Genetic_Algorithm
from openmdao.drivers.amiego_util.kriging import KrigingSurrogate
from openmdao.utils.concurrent import concurrent_eval, concurrent_eval_lb
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import FakeComm

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


def snopt_opt(objfun, desvar, lb, ub, ncon, title=None, options=None,
              sens='FD', jac=None):
    """ Wrapper function for running a SNOPT optimization through
    pyoptsparse."""

    if OPTIMIZER:
        from pyoptsparse import Optimization
    else:
        raise(RuntimeError, 'Need pyoptsparse to run the SNOPT sub optimizer.')

    opt_prob = Optimization(title, objfun, comm=FakeComm())

    ndv = len(desvar)

    opt_prob.addVarGroup('x', ndv, type='c', value=desvar.flatten(), lower=lb.flatten(),
                         upper=ub.flatten())
    opt_prob.addConGroup('con', ncon, upper=np.zeros((ncon)))#, linear=True, wrt='x',
                         #jac={'x' : jac})
    opt_prob.addObj('obj')

    # Fall back on SLSQP if SNOPT isn't there
    _tmp = __import__('pyoptsparse', globals(), locals(), [OPTIMIZER], 0)
    opt = getattr(_tmp, OPTIMIZER)()

    if options:
        for name, value in iteritems(options):
            opt.setOption(name, value)

    if OPTIMIZER == 'SNOPT':
        opt.setOption('Major iterations limit', 100)
        opt.setOption('Verify level', -1)
        opt.setOption('iSumm', 0)
        opt.setOption('iPrint', 0)
    elif OPTIMIZER == 'SLSQP':
        opt.setOption('MAXIT', 100)
    elif OPTIMIZER == 'CONMIN':
        opt.setOption('ITMAX', 100)

    sol = opt(opt_prob, sens=sens, sensStep=1.0e-6)
    #print(sol)

    x = sol.getDVs()['x']
    f = sol.objectives['obj'].value
    success_flag = sol.optInform['value'] < 2

    return x, f, success_flag


def snopt_opt2(objfun, desvar, lb, ub, title=None, options=None,
              sens='FD', jac=None):
    """ Wrapper function for running a SNOPT optimization through
    pyoptsparse."""

    if OPTIMIZER:
        from pyoptsparse import Optimization
    else:
        raise(RuntimeError, 'Need pyoptsparse to run the SNOPT sub optimizer.')

    opt_prob = Optimization(title, objfun, comm=FakeComm())

    ndv = len(desvar)

    opt_prob.addVarGroup('x', ndv, type='c', value=desvar.flatten(), lower=lb.flatten(),
                         upper=ub.flatten())
    opt_prob.addObj('obj')

    # Fall back on SLSQP if SNOPT isn't there
    _tmp = __import__('pyoptsparse', globals(), locals(), [OPTIMIZER], 0)
    opt = getattr(_tmp, OPTIMIZER)()


    if options:
        for name, value in iteritems(options):
            opt.setOption(name, value)

    if OPTIMIZER == 'SNOPT':
        opt.setOption('Major iterations limit', 100)
        opt.setOption('Verify level', -1)
        opt.setOption('iSumm', 0)
        opt.setOption('iPrint', 0)
    elif OPTIMIZER == 'SLSQP':
        opt.setOption('MAXIT', 100)
    elif OPTIMIZER == 'CONMIN':
        opt.setOption('ITMAX', 100)

    sol = opt(opt_prob, sens=sens, sensStep=1.0e-6)
    #print(sol)

    x = sol.getDVs()['x']
    f = sol.objectives['obj'].value
    success_flag = sol.optInform['value'] < 2
    msg = sol.optInform['text']

    return x, f, success_flag, msg


class Branch_and_Bound(Driver):
    """ Class definition for the Branch_and_Bound driver. This driver can only be run
    plugged into the AMIEGO driver.

    This is the branch and bound algorithm that maximizes the constrained
    expected improvement function and returns an integer infill point. The
    algorithm uses the relaxation techniques proposed by Jones et.al. on
    their paper on EGO,1998. This enables the algorithm to use any
    gradient-based approach to obtain a global solution. Also, to satisfy the
    integer constraints, a new branching scheme has been implemented.
    """

    def __init__(self):
        """Initialize the Branch_and_Bound driver."""

        super(Branch_and_Bound, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = False
        self.supports['mixed_integer'] = False

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = KrigingSurrogate

        # TODO: is 1e-3 the right and best value for ftol?

        # Options
        opt = self.options
        opt.declare('active_tol', 1.0e-6, lower=0.0,
                    desc='Tolerance (2-norm) for triggering active set '
                    'reduction.')
        opt.declare('atol', 0.1, lower=0.0,
                    desc='Absolute tolerance (inf-norm) of upper minus '
                    'lower bound for termination.')
        opt.declare('con_tol', 1.0e-6, lower=0.0,
                    desc='Constraint thickness.')
        opt.declare('concave_EI', False,
                    desc='Set to True to apply a transformation to make the '
                    'objective function concave.')
        opt.declare('disp', True,
                    desc='Set to False to prevent printing of iteration '
                    'messages.')
        opt.declare('ftol', 1.0e-4, lower=0.0,
                    desc='Absolute tolerance for sub-optimizations.')
        opt.declare('maxiter', 100000, lower=0.0,
                    desc='Maximum number of iterations.')
        opt.declare('penalty_factor', 0.0,
                    desc='Penalty weight on objective using radial functions.')
        opt.declare('penalty_width', 0.5,
                    desc='Penalty width on objective using radial functions.')
        opt.declare('trace_iter', 3,
                    desc='Number of generations to trace back for ubd.')
        opt.declare('trace_iter_max', 5,
                    desc='Maximum number of generations to trace back for ubd.')
        opt.declare('maxiter_ubd', 10000,
                    desc='Number of generations ubd stays the same')
        opt.declare('use_surrogate', False,
                    desc='Use surrogate model for the optimization. Training '
                    'data must be supplied.')
        opt.declare('local_search', 0, values=[0, 1, 2],
                    desc='Local search type. Set to 0 for GA, 1 for LHS, 2 for LHS + SQP '
                    '(Default = 0)')

        # Initial Sampling
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

        self.dvs = []
        self.size = 0
        self.idx_cache = {}
        self.obj_surrogate = None

        # Amiego will set this to True if we have found a minimum.
        self.eflag_MINLPBB = False

        # Amiego retrieves optimal design and optimum upon completion.
        self.xopt = None
        self.fopt = None

        # Switch between pyoptsparse and scipy/slsqp
        self.pyopt = True

        # Declare stuff we need to pass to objective callback functions
        self.current_surr = None

        # Experimental Options. TODO: could go into Options
        self.load_balance = True
        self.aggressive_splitting = False

    def _setup_driver(self, problem, assemble_var_info=True):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        assemble_var_info : bool
            If True, then gather all the designvars, objectives, and constraints from the model.
        """
        super(Branch_and_Bound, self)._setup_driver(problem, assemble_var_info)

        # Size our design variables.
        j = 0
        for name, val in iteritems(self.get_design_var_values()):
            self.dvs.append(name)
            size = len(val)
            self.idx_cache[name] = (j, j+size)
            j += size
        self.size = j

        # Lower and Upper bounds
        self.xI_lb = np.empty((self.size))
        self.xI_ub = np.empty((self.size))
        dv_dict = self._designvars
        for var in self.dvs:
            i, j = self.idx_cache[var]
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

    def run(self, problem):
        """
        Execute the Branch_and_Bound method..

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem
        obj_surrogate = self.obj_surrogate
        n_i = self.size
        atol = self.options['atol']
        ftol = self.options['ftol']
        disp = self.options['disp']
        maxiter = self.options['maxiter']
        maxiter_ubd = self.options['maxiter_ubd']

        self.iter_count = 1

        if obj_surrogate:
            obj_surrogate.p = 2
            obj_surrogate.y_best = np.min(obj_surrogate.Y)
            #TODO If you are using Y value from obj_surrogate, no need to normalize in the calc_conEI_norm function

        #----------------------------------------------------------------------
        # Step 1: Initialize
        #----------------------------------------------------------------------

        terminate = False
        num_des = len(self.xI_lb)
        node_num = 0
        itercount = 0
        ubd_count = 0

        # Initial B&B bounds are infinite.
        UBD = np.inf
        LBD = -np.inf
        LBD_prev =- np.inf

        # copy our desvars' user specified upper and lower bounds
        # FIXME: is this copy really needed here since we copy these again inside the loop?
        xL_iter = self.xI_lb.copy()
        xU_iter = self.xI_ub.copy()

        num_init_sam = num_des
        init_sam = lhs(num_des, samples=num_init_sam, criterion='center')
        for ii in range(num_init_sam):
            xopt_ii = np.round(xL_iter + init_sam[ii]*(xU_iter - xL_iter)).reshape(num_des)
            fopt_ii = self.objective_callback(xopt_ii)
            if fopt_ii < UBD:
                self.eflag_MINLPBB = True
                UBD = fopt_ii
                fopt = fopt_ii
                xopt = xopt_ii

        # This stuff is just for printing.
        par_node = 0

        # Active set fields: (Updated!)
        #     Aset = [[NodeNumber, lb, ub, LBD, UBD, nodeHist], [], ..]
        # Each node is a list.
        active_set = []
        nodeHist = nodeHistclass()
        UBD_term = UBD

        comm = problem.model.comm
        if self.load_balance:

            # Master/Worker config
            n_proc = comm.size - 1
            if n_proc < 2:
                comm = None
                n_proc = 1

        else:

            # Each proc has its own jobs
            n_proc = comm.size
            if n_proc < 2:
                comm = None

        # Initial node. This is the data structure we pass into the
        # concurrent evaluator. TODO: wonder if we can clean this up.
        if self.aggressive_splitting:

            # Initial number of nodes based on number of available procs
            args = init_nodes(n_proc, xL_iter, xU_iter, par_node, LBD_prev, LBD,
                              UBD, fopt, xopt, nodeHist, ubd_count)
        else:

            # Start with 1 node.
            args = [(xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                xopt, node_num, nodeHist, ubd_count)]

        #Evaluate the concavity factor
        self.con_fac = concave_factor(xL_iter,xU_iter,obj_surrogate)
        # Main Loop
        while not terminate:

            # Branch and Bound evaluation of a set of nodes, starting with the initial one.
            # When executed in serial, only a single node is evaluted.
            cases = [(arg, None) for arg in args]

            if self.load_balance:
                results = concurrent_eval_lb(self.evaluate_node, cases,
                                             comm, broadcast=True)
            else:
                results = concurrent_eval(self.evaluate_node, cases,
                                          comm, allgather=True)

            itercount += len(args)

            if UBD < -1.0e-3:
                ubd_count += len(args)
            # Put all the new nodes into active set.
            for result in results:

                # Print the traceback if it fails
                if not result[0]:
                    print(result[1])

                new_UBD, new_fopt, new_xopt, new_nodes = result[0]

                # Save stats for the best case.
                if new_UBD < UBD:
                    UBD = new_UBD
                    fopt = new_fopt
                    xopt = new_xopt
                if abs(new_UBD-UBD_term)>0.001: #Look for substantial change in UBD to reset the counter
                    ubd_count = 1
                    UBD_term = new_UBD

                # TODO: Should we extend the active set with all the cases we
                # ran, or just the best one. All for now.
                active_set.extend(new_nodes)
                node_num += len(new_nodes)

            # Update active set: Removes all nodes worse than the best new node.
            if len(active_set) >= 1:
                active_set = update_active_set(active_set, UBD)

            # Termination
            if len(active_set) >= 1:
                # Update LBD and select the current rectangle

                args = []

                # Grab the best nodes, as many as we have processors.
                n_nodes = np.min((n_proc, len(active_set)))
                for j in range(n_nodes):

                    # a. Set LBD as lowest in the active set
                    all_LBD = [item[3] for item in active_set]
                    LBD = min(all_LBD)

                    ind_LBD = all_LBD.index(LBD)
                    LBD_prev = LBD

                    # b. Select the lowest LBD node as the current node
                    par_node, xL_iter, xU_iter, _, _, nodeHist = active_set[ind_LBD]
                    self.iter_count += 1

                    args.append((xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                                 xopt, node_num, nodeHist, ubd_count))

                    # c. Delete the selected node from the Active set of nodes
                    del active_set[ind_LBD]

                    #--------------------------------------------------------------
                    #Step 7: Check for convergence
                    #--------------------------------------------------------------
                    diff = np.abs(UBD - LBD)
                    if diff < atol:
                        terminate = True
                        if disp:
                            print("="*85)
                            print("Terminating! Absolute difference between the upper " + \
                                  "and lower bound is below the tolerence limit.")
            else:
                terminate = True
                if disp:
                    print("="*85)
                    print("Terminating! No new node to explore.")
                    print("Max Node", node_num)

            if itercount > maxiter or ubd_count > maxiter_ubd:
                terminate = True

        # Finalize by putting optimal value back into openMDAO
        self.xopt = xopt
        self.fopt = fopt

    def evaluate_node(self, xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD,
                      fopt, xopt, node_num, nodeHist, ubd_count):
        """
        Perform Branch and Bound step on a single node.

        This function encapsulates the portion of the code that runs in parallel.
        """

        if OPTIMIZER == 'SNOPT':
            options = {'Major optimality tolerance' : 1.0e-5}
        elif OPTIMIZER == 'SLSQP':
            options = {'ACC' : 1.0e-5}
        elif OPTIMIZER == 'CONMIN':
            options = {'DABFUN' : 1.0e-5}

        active_tol = self.options['active_tol']
        local_search = self.options['local_search']
        disp = self.options['disp']
        trace_iter = self.options['trace_iter']
        trace_iter_max = self.options['trace_iter_max']

        obj_surrogate = self.obj_surrogate
        num_des = len(self.xI_lb)

        new_nodes = []

        #Keep this to 0.49 to always round towards bottom-left
        xloc_iter = np.round(xL_iter + 0.49*(xU_iter - xL_iter))
        floc_iter = self.objective_callback(xloc_iter)
        efloc_iter = True

        # Genetic Algorithm
        if local_search == 0:

            ga = Genetic_Algorithm(calc_conEI_norm)

            bits = np.zeros((num_des, 1), dtype = int)
            bits = np.ceil(np.log2(xU_iter - xL_iter + 1))
            bits[bits<=0] =1
            vub_vir = (2**bits - 1) + xL_iter
            if nodeHist.priority_flag == 1:
                max_gen=300
                mfac=6
            else:
                max_gen=200
                mfac=4
            L = np.sum(bits)
            pop_size = mfac*L

            t0 = time()
            xloc_iter_new, floc_iter_new, nfit = \
                ga.execute_ga(xL_iter, vub_vir, bits, pop_size, max_gen, obj_surrogate, xU_iter)
            t_GA = time() - t0

            if floc_iter_new < floc_iter:
                floc_iter = floc_iter_new
                xloc_iter = xloc_iter_new

        # LHS Sampling or SNOPT
        else:
            #TODO Future research on sampling here
            num_samples = np.round(np.max([10, np.min([50, num_des/nodeHist.priority_flag])])) #TODO Future research
            init_sam_node = lhs(num_des, samples=num_samples, criterion='center')
            l_succ = 0
            t_GA = 0.

            for ii in range(int(num_samples)):
                xloc_iter_new = np.round(xL_iter + init_sam_node[ii]*(xU_iter - xL_iter))
                floc_iter_new = self.objective_callback(xloc_iter_new)

                # SNOPT
                if local_search == 2:
                    # TODO: did we lose a tol check here?
                    # active_tol: #Perform at non-flat starting point
                    if np.abs(floc_iter_new) > -np.inf:
                        l_succ += 1
                        #--------------------------------------------------------------
                        #Step 2: Obtain a local solution
                        #--------------------------------------------------------------
                        # Using a gradient-based method here.
                        # TODO: Make it more pluggable.
                        def _objcall(dv_dict):
                            """ Callback function"""
                            fail = 0
                            x = dv_dict['x']
                            # Objective
                            func_dict = {}
                            confac_flag = False
                            func_dict['obj'] = self.objective_callback(x)[0]
                            return func_dict, fail

                        xC_iter = xloc_iter_new
                        opt_x, opt_f, succ_flag, msg = snopt_opt2(_objcall, xC_iter, xL_iter,
                                                                  xU_iter, title='LocalSearch',
                                                                  options=options)

                        xloc_iter_new = np.round(np.asarray(opt_x).flatten())
                        floc_iter_new = self.objective_callback(xloc_iter_new)

                if floc_iter_new < floc_iter:
                    floc_iter = floc_iter_new
                    xloc_iter = xloc_iter_new

        # Do some prechecks before commencing for partitioning.
        ubdloc_best = nodeHist.ubdloc_best
        if nodeHist.ubdloc_best > floc_iter + 1.0e-6:
            ubd_track = np.concatenate((nodeHist.ubd_track, np.array([0])), axis=0)
            ubdloc_best = floc_iter
        else:
            ubd_track = np.concatenate((nodeHist.ubd_track, np.array([1])), axis=0)

        # diff_LBD = abs(LBD_prev - LBD_NegConEI)
        if len(ubd_track) >= trace_iter_max or \
           (len(ubd_track) >= trace_iter and np.sum(ubd_track[-trace_iter:]) == 0):
            # TODO : Did we lose ths? -> #and UBD<=-1.0e-3:
            child_info = np.array([[par_node, np.inf, floc_iter], [par_node, np.inf, floc_iter]])

            #Fathomed due to no change in UBD_loc for 'trace_iter' generations
            dis_flag = ['Y', 'Y']

        else:
            #--------------------------------------------------------------
            # Step 3: Partition the current rectangle as per the new
            # branching scheme.
            #--------------------------------------------------------------
            child_info = np.zeros([2, 3])
            dis_flag = [' ', ' ']

            # Choose
            l_iter = (xU_iter - xL_iter).argmax()

            if xloc_iter[l_iter] < xU_iter[l_iter]:
                delta = 0.5 #0<delta<1
            else:
                delta = -0.5 #-1<delta<0

            for ii in range(2):
                lb = xL_iter.copy()
                ub = xU_iter.copy()
                if ii == 0:
                    ub[l_iter] = np.floor(xloc_iter[l_iter]+delta)
                elif ii == 1:
                    lb[l_iter] = np.ceil(xloc_iter[l_iter]+delta)

                if np.linalg.norm(ub - lb) > active_tol: #Not a point
                    #--------------------------------------------------------------
                    # Step 4: Obtain an LBD of f in the newly created node
                    #--------------------------------------------------------------
                    S4_fail = False
                    x_comL, x_comU, Ain_hat, bin_hat = gen_coeff_bound(lb, ub, obj_surrogate)
                    sU, eflag_sU = self.maximize_S(x_comL, x_comU, Ain_hat, bin_hat,
                                                   obj_surrogate)

                    if eflag_sU:
                        yL, eflag_yL = self.minimize_y(x_comL, x_comU, Ain_hat, bin_hat,
                                                       obj_surrogate)

                        if eflag_yL:
                            NegEI = calc_conEI_norm([], obj_surrogate, SSqr=sU, y_hat=yL)
                        else:
                            S4_fail = True
                    else:
                        S4_fail = True

                    # Convex approximation failed!
                    if S4_fail:
                        if efloc_iter:
                            LBD_NegConEI = LBD_prev
                        else:
                            LBD_NegConEI = np.inf
                        dis_flag[ii] = 'F'
                    else:
                        LBD_NegConEI = max(NegEI, LBD_prev)

                    #--------------------------------------------------------------
                    # Step 5: Store any new node inside the active set that has LBD
                    # lower than the UBD.
                    #--------------------------------------------------------------

                    priority_flag = 0
                    if LBD_NegConEI < np.inf and LBD_prev < np.inf:
                        if np.abs((LBD_prev - LBD_NegConEI)/LBD_prev) < 0.005:
                            priority_flag = 1

                    nodeHist_new = nodeHistclass()
                    nodeHist_new.ubd_track = ubd_track
                    nodeHist_new.ubdloc_best = ubdloc_best
                    nodeHist_new.priority_flag = priority_flag

                    if LBD_NegConEI < UBD - 1.0e-6:
                        node_num += 1
                        new_node = [node_num, lb, ub, LBD_NegConEI, floc_iter, nodeHist_new]
                        new_nodes.append(new_node)
                        child_info[ii] = np.array([node_num, LBD_NegConEI, floc_iter])
                    else:
                        child_info[ii] = np.array([par_node, LBD_NegConEI, floc_iter])
                        dis_flag[ii] = 'X' #Flag for child created but not added to active set (fathomed)
                else:
                    if ii == 1:
                        xloc_iter = ub
                        floc_iter = self.objective_callback(xloc_iter)
                    child_info[ii] = np.array([par_node, np.inf, floc_iter])
                    dis_flag[ii] = 'x' #Flag for No child created

        #Update the active set whenever better solution found
        if floc_iter < UBD:
            UBD = floc_iter
            fopt = floc_iter
            xopt = xloc_iter.copy().reshape(num_des)

        if disp:
            if (self.iter_count-1) % 25 == 0:
                # Display output in a tabular format
                print("="*95)
                print("%19s%12s%14s%21s" % ("Global", "Parent", "Child1", "Child2"))
                template = "%s%8s%10s%8s%9s%11s%10s%11s%11s%11s"
                print(template % ("Iter", "LBD", "UBD", "Node", "Node1", "LBD1",
                                  "Node2", "LBD2", "Flocal", "GA time"))
                print("="*95)
            template = "%3d%10.2f%10.2f%6d%8d%1s%13.2f%8d%1s%13.2f%9.2f%9.2f"
            print(template % (self.iter_count, LBD, UBD, par_node, child_info[0, 0],
                              dis_flag[0], child_info[0, 1], child_info[1, 0],
                              dis_flag[1], child_info[1, 1], child_info[1, 2], t_GA))

        return UBD, fopt, xopt, new_nodes

    def objective_callback(self, xI, con_EI=False):
        """ Callback for main problem evaluation."""
        obj_surrogate = self.obj_surrogate

        # When run under AMEIGO, objective is the expected improvment
        # function with modifications to make it concave.

        X = obj_surrogate.X
        k = np.shape(X)[1]

        # Normalized as per the convention in Kriging of openmdao
        xval = (xI - obj_surrogate.X_mean.flatten())/obj_surrogate.X_std.flatten()

        NegEI = calc_conEI_norm(xval, obj_surrogate)

        P = 0.0
        f = NegEI + P

        #print(xI, f)
        return f

    def maximize_S(self, x_comL, x_comU, Ain_hat, bin_hat, surrogate):
        """This method finds an upper bound to the SigmaSqr Error, and scales
        up 'r' to provide a smooth design space for gradient-based approach.
        """

        if OPTIMIZER == 'SNOPT':
            options = {'Major optimality tolerance' : 1.0e-5}
        elif OPTIMIZER == 'SLSQP':
            options = {'ACC' : 1.0e-5}
        elif OPTIMIZER == 'CONMIN':
            options = {'DABFUN' : 1.0e-5}

        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        X = surrogate.X

        n, k = X.shape
        one = np.ones([n, 1])

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = 0.0
        xhat_comU[k:] = 1.0

        # Calculate the convexity factor alpha
        rL = x_comL[k:]
        rU = x_comU[k:]

        dr_drhat = np.zeros([n, n])
        for ii in range(n):
            dr_drhat[ii, ii] = rU[ii, 0] - rL[ii, 0]

        T2_num = np.dot(np.dot(R_inv, one), np.dot(R_inv, one).T)
        T2_den = np.dot(one.T, np.dot(R_inv, one))
        d2S_dr2 = 2.0*SigmaSqr*(R_inv - (T2_num/T2_den))
        H_hat = np.dot(np.dot(dr_drhat, d2S_dr2), dr_drhat.T)

        # Use Gershgorin's circle theorem to find a lower bound of the
        # min eigen value of the hessian
        eig_lb = np.zeros([n, 1])
        for ii in range(n):
            dia_ele = H_hat[ii, ii]
            sum_rw = 0.0
            sum_col = 0.0
            for jj in range(n):
                if ii != jj:
                    sum_rw += np.abs(H_hat[ii,jj])
                    sum_col += np.abs(H_hat[jj,ii])

                eig_lb[ii] = dia_ele - np.min(np.array([sum_rw, sum_col]))

        eig_min = np.min(eig_lb)
        alpha = np.max(np.array([0.0, -0.5*eig_min]))

        # Just storing it here to pull it out in the callback?
        surrogate._alpha = alpha

        # Maximize S
        x0 = 0.5*(xhat_comL + xhat_comU)
        #bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in range(len(xhat_comL))]

        if self.pyopt:

            self.x_comL = x_comL
            self.x_comU = x_comU
            self.xhat_comL = xhat_comL
            self.xhat_comU = xhat_comU
            self.Ain_hat = Ain_hat
            self.bin_hat = bin_hat
            self.current_surr = surrogate

            opt_x, opt_f, succ_flag = snopt_opt(self.calc_SSqr_convex, x0, xhat_comL,
                                                xhat_comU, len(bin_hat),
                                                title='Maximize_S',
                                                options=options,
                                                jac=Ain_hat,
                                                ) #sens=self.calc_SSqr_convex_grad)

            Neg_sU = opt_f
            # if not succ_flag:
            #     eflag_sU = False
            # else:
            #     eflag_sU = True
            eflag_sU = True
            tol = self.options['con_tol']
            for ii in range(2*n):
                if np.dot(Ain_hat[ii, :], opt_x) > (bin_hat[ii ,0] + tol):
                    eflag_sU = False
                    break

        else:
            optResult = minimize(self.calc_SSqr_convex_old, x0,
                                 args=(x_comL, x_comU, xhat_comL, xhat_comU, surrogate),
                                 method='SLSQP', constraints=[], bounds=bnds,
                                 options={'ftol' : self.options['ftol'],
                                          'maxiter' : 100})

            Neg_sU = optResult.fun
            if not optResult.success:
                eflag_sU = False
            else:
                eflag_sU = True
                tol = self.options['con_tol']
                for ii in range(2*n):
                    if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii ,0] + tol):
                        eflag_sU = False
                        break

        sU = - Neg_sU
        return sU, eflag_sU

    def calc_SSqr_convex(self, dv_dict):
        """ Callback function for minimization of mean squared error."""
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU
        xhat_comL = self.xhat_comL
        xhat_comU = self.xhat_comU

        X = surrogate.X
        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        alpha = surrogate._alpha

        n, k = X.shape

        one = np.ones([n, 1])

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = x_com[k:].reshape(n, 1)

        r = rL + rhat*(rU - rL)
        rhat_L = xhat_comL[k:]
        rhat_U = xhat_comU[k:]

        term0 = np.dot(R_inv, r)
        term1 = -SigmaSqr*(1.0 - r.T.dot(term0) + \
        ((1.0 - one.T.dot(term0))**2/(one.T.dot(np.dot(R_inv, one)))))

        term2 = alpha*(rhat-rhat_L).T.dot(rhat-rhat_U)
        S2 = term1 + term2

        # Objectives
        func_dict = {}
        func_dict['obj'] = S2[0, 0]

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        func_dict['con'] = np.dot(Ain_hat, x_com) - bin_hat.flatten()
        #print('x', dv_dict)
        #print('obj', func_dict['obj'])
        return func_dict, fail

    def calc_SSqr_convex_grad(self, dv_dict, func_dict):
        """ Callback function for gradient of mean squared error."""
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU
        xhat_comL = self.xhat_comL
        xhat_comU = self.xhat_comU

        X = surrogate.X
        R_inv = surrogate.R_inv
        SigmaSqr = surrogate.SigmaSqr
        alpha = surrogate._alpha

        n, k = X.shape
        nn = len(x_com)

        one = np.ones([n, 1])

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = x_com[k:].reshape(n, 1)

        r = rL + rhat*(rU - rL)
        rhat_L = xhat_comL[k:]
        rhat_U = xhat_comU[k:]

        dr_drhat = np.diag((rU-rL).flat)

        term0 = np.dot(R_inv, r) #This should be nx1 vector
        term1 = ((1.0 - one.T.dot(term0))/(one.T.dot(np.dot(R_inv, one))))*np.dot(R_inv, one) #This should be nx1 vector
        term = 2.0*SigmaSqr*(term0 + term1) #This should be nx1 vector

        #zdterm1a = (r.T.dot(R_inv) + r.T.dot(R_inv.T))
        #zdterm1b = 2.0*((1.0 - one.T.dot(term0))*np.sum(R_inv, 0)/(one.T.dot(np.dot(R_inv, one))))
        #zdterm1 = SigmaSqr*(zdterm1a.T + dr_drhat.dot(zdterm1b.T))

        dterm1 = np.dot(dr_drhat, term) #This should be nx1 vector
        dterm2 = alpha*(2.0*rhat - rhat_L - rhat_U) #This should be nx1 vector

        dobj_dr = (dterm1 + dterm2).T #This should be 1xn vector

        # Objectives
        sens_dict = OrderedDict()
        sens_dict['obj'] = OrderedDict()
        sens_dict['obj']['x'] = np.zeros((1, nn))
        sens_dict['obj']['x'][:, k:] = dobj_dr

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        sens_dict['con'] = OrderedDict()
        sens_dict['con']['x'] = Ain_hat

        #print('obj deriv', sens_dict['obj']['x'] )
        #print('con deriv', sens_dict['con']['x'])
        return sens_dict, fail

    def minimize_y(self, x_comL, x_comU, Ain_hat, bin_hat, surrogate):

        if OPTIMIZER == 'SNOPT':
            options = {'Major optimality tolerance' : 1.0e-8}
        elif OPTIMIZER == 'SLSQP':
            options = {'ACC' : 1.0e-8}
        elif OPTIMIZER == 'CONMIN':
            options = {'DABFUN' : 1.0e-8}

        # 1- Formulates y_hat as LP (weaker bound)
        # 2- Uses non-convex relaxation technique (stronger bound) [Future release]
        app = 1

        X = surrogate.X
        n, k = X.shape

        xhat_comL = x_comL.copy()
        xhat_comU = x_comU.copy()
        xhat_comL[k:] = 0.0
        xhat_comU[k:] = 1.0

        if app == 1:
            x0 = 0.5*(xhat_comL + xhat_comU)
            #bnds = [(xhat_comL[ii], xhat_comU[ii]) for ii in range(len(xhat_comL))]

        if self.pyopt:

            self.x_comL = x_comL
            self.x_comU = x_comU
            self.Ain_hat = Ain_hat
            self.bin_hat = bin_hat
            self.current_surr = surrogate

            opt_x, opt_f, succ_flag = snopt_opt(self.calc_y_hat_convex, x0, xhat_comL,
                                                xhat_comU, len(bin_hat),
                                                title='minimize_y',
                                                options=options,
                                                jac=Ain_hat)

            yL = opt_f
            # if not succ_flag:
            #     eflag_yL = False
            # else:
            #     eflag_yL = True
            eflag_yL = True
            tol = self.options['con_tol']
            for ii in range(2*n):
                if np.dot(Ain_hat[ii, :], opt_x) > (bin_hat[ii, 0] + tol):
                    eflag_yL = False
                    break

        else:
            optResult = minimize(self.calc_y_hat_convex_old, x0,
                                 args=(x_comL, x_comU, surrogate), method='SLSQP',
                                 constraints=[], bounds=bnds,
                                 options={'ftol' : self.options['ftol'],
                                          'maxiter' : 100})

            yL = optResult.fun
            if not optResult.success:
                eflag_yL = False
            else:
                eflag_yL = True
                tol = self.options['con_tol']
                for ii in range(2*n):
                    if np.dot(Ain_hat[ii, :], optResult.x) > (bin_hat[ii, 0] + tol):
                        eflag_yL = False
                        break

        return yL, eflag_yL

    def calc_y_hat_convex_old(self, x_com, *param):
        x_comL = param[0]
        x_comU = param[1]
        surrogate = param[2]

        X = surrogate.X
        c_r = surrogate.c_r
        mu = surrogate.mu
        n, k = X.shape

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)
        r = rL + rhat*(rU - rL)

        y_hat = mu + np.dot(r.T, c_r)
        return y_hat[0, 0]

    def calc_y_hat_convex(self, dv_dict):
        fail = 0

        x_com = dv_dict['x']
        surrogate = self.current_surr
        x_comL = self.x_comL
        x_comU = self.x_comU

        X = surrogate.X
        c_r = surrogate.c_r
        mu = surrogate.mu
        n, k = X.shape

        rL = x_comL[k:]
        rU = x_comU[k:]
        rhat = np.array([x_com[k:]]).reshape(n, 1)
        r = rL + rhat*(rU - rL)

        y_hat = mu + np.dot(r.T, c_r)

        # Objective
        func_dict = {}
        func_dict['obj'] = y_hat[0, 0]

        # Constraints
        Ain_hat = self.Ain_hat
        bin_hat = self.bin_hat

        func_dict['con'] = np.dot(Ain_hat, x_com) - bin_hat.flatten()
        #print('x', dv_dict)
        #print('obj', func_dict['obj'])
        return func_dict, fail

def update_active_set(active_set, ubd):
    """ Remove variables from the active set data structure if their current
    upper bound exceeds the given value.

    Args
    ----
    active_set : list of lists of floats
        Active set data structure of form [[NodeNumber, lb, ub, LBD, UBD], [], ..]
    ubd : float
        Maximum for bounds test.

    Returns
    -------
    list of list of floats
        New active_set
    """
    return [a for a in active_set if a[3] < ubd]


def gen_coeff_bound(xI_lb, xI_ub, surrogate):
    """This function generates the upper and lower bound of the artificial
    variable r and the coefficients for the linearized under estimator
    constraints. The version accepts design bound in the original design
    space, converts it to normalized design space.
    """

    #Normalized as per Openmdao kriging model
    xL_hat = (xI_lb - surrogate.X_mean.flatten())/surrogate.X_std.flatten()
    xU_hat = (xI_ub - surrogate.X_mean.flatten())/surrogate.X_std.flatten()

    rL, rU = interval_analysis(xL_hat, xU_hat, surrogate)

    # Combined design variables for supbproblem
    num = len(xL_hat) + len(rL)
    x_comL = np.append(xL_hat, rL).reshape(num, 1)
    x_comU = np.append(xU_hat, rU).reshape(num, 1)

    # Coefficients of the linearized constraints of the subproblem
    Ain_hat, bin_hat = lin_underestimator(x_comL, x_comU, surrogate)

    return x_comL, x_comU, Ain_hat, bin_hat


def interval_analysis(lb_x, ub_x, surrogate):
    """ The module predicts the lower and upper bound of the artificial
    variable 'r' from the bounds of the design variable x r is related to x
    by the following equation:

    r_i = exp(-sum(theta_h*(x_h - x_h_i)^2))

    """
    p = surrogate.p

    if p % 2 == 0:
        X = surrogate.X
        thetas = surrogate.thetas
        n, k = X.shape

        t3L = np.empty([n, k])
        t3U = np.empty([n, k])

        t1L = lb_x - X
        t1U = ub_x - X

        fac1 = t1L*t1L
        fac2 = t1L*t1U
        fac3 = t1U*t1U

        for i in range(n):
            for h in range(k):

                fact = np.array([fac1[i, h], fac2[i, h], fac3[i, h]])
                t2L = np.max(np.array([0, np.min(fact)]))
                t2U = np.max(np.array([0, np.max(fact)]))

                fact = np.array([-thetas[h]*t2L, -thetas[h]*t2U])
                t3L[i, h] = np.min(fact)
                t3U[i, h] = np.max(fact)

        lb_r = np.exp(np.sum(t3L, axis=1))
        ub_r = np.exp(np.sum(t3U, axis=1))
    else:
        print("\nWarning! Value of p should be 2. Cannot perform interval analysis")
        print("\nReturing global bound of the r variable")
        lb_r = np.zeros([n, k])
        ub_r = np.zeros([n, k])

    return lb_r, ub_r


def lin_underestimator(lb, ub, surrogate):
    X = surrogate.X
    thetas = surrogate.thetas
    p = surrogate.p
    n, k = X.shape

    lb_x = lb[:k]; ub_x = ub[:k]
    lb_r = lb[k:]; ub_r = ub[k:]

    a1_hat = np.zeros([n, n]); a3_hat = np.zeros([n, n])
    a2 = np.zeros([n, k]); a4 = np.zeros([n, k])
    b2 = np.zeros([n, k]); b4 = np.zeros([n, k])
    b1_hat = np.zeros([n, 1]); b3_hat = np.zeros([n, 1])

    for i in range(n):
        #T1: Linearize under-estimator of ln[r_i] = a1*r[i] + b1
        if ub_r[i] <= lb_r[i]:
            a1 = 0.0
        else:
            a1 = ((np.log(ub_r[i]) - np.log(lb_r[i]))/(ub_r[i] - lb_r[i]))

        b1 = np.log(ub_r[i]) - a1*ub_r[i]
        a1_hat[i, i] = a1*(ub_r[i] - lb_r[i])
        b1_hat[i] = a1*lb_r[i] + b1

        #T3: Linearize under-estimator of -ln[r_i] = a3*r[i] + b3
        r_m_i = (lb_r[i] + ub_r[i])/2.0
        a3 = -1.0/r_m_i
        b3 = -np.log(r_m_i) - a3*r_m_i
        a3_hat[i,i] = a3*(ub_r[i] - lb_r[i])
        b3_hat[i] = a3*lb_r[i] + b3

        for h in range(k):
            #T2: Linearize under-estimator of thetas_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            x_m_h = (ub_x[h] + lb_x[h])/2.0
            a2[i,h] = p*thetas[h]*(x_m_h - X[i,h])**(p-1.0)
            yy = thetas[h]*(x_m_h - X[i,h])**p
            b2[i,h] = -a2[i,h]*x_m_h + yy

            #T4: Linearize under-estimator of -theta_h*(x_h - X_h_i)^2 = a4[i,h]*x_h[h] + b4[i,h]
            yy2 = -thetas[h]*(ub_x[h] - X[i, h])**p
            yy1 = -thetas[h]*(lb_x[h] - X[i, h])**p

            if ub_x[h] <= lb_x[h]:
                a4[i,h] = 0.0
            else:
                a4[i,h] = (yy2 - yy1)/(ub_x[h] - lb_x[h])

            b4[i, h] = -a4[i, h]*lb_x[h] + yy1

    Ain1 = np.concatenate((a2, a4), axis=0)
    Ain2 = np.concatenate((a1_hat, a3_hat), axis=0)
    Ain_hat = np.concatenate((Ain1, Ain2), axis=1)
    bin_hat = np.concatenate((-(b1_hat + np.sum(b2, axis=1).reshape(n, 1)),
                              -(b3_hat + np.sum(b4, axis=1).reshape(n, 1))), axis=0)

    return Ain_hat, bin_hat

def calc_conEI_norm(xval, obj_surrogate, SSqr=None, y_hat=None):
    """This function evaluates the expected improvement in the normalized
    design space.
    """
    # Ensure y_min is the minimum of the y used to train the surrogates (i.e centered/scaled/normalized y)
    y_min = obj_surrogate.y_best

    if SSqr is None:
        X = obj_surrogate.X
        c_r = obj_surrogate.c_r
        thetas = obj_surrogate.thetas
        SigmaSqr = obj_surrogate.SigmaSqr
        R_inv = obj_surrogate.R_inv
        mu = obj_surrogate.mu
        p = obj_surrogate.p

        n = np.shape(X)[0]
        one = np.ones([n, 1])

        r = np.exp(-np.sum(thetas.T*(xval - X)**p, 1)).reshape(n, 1)

        y_hat = mu + np.dot(r.T, c_r)
        term0 = np.dot(R_inv, r)

        # Note: This dot product with one stuff seems to be faster than np.sum for these small
        # sized matrices.
        SSqr = SigmaSqr*(1.0 - r.T.dot(term0) + \
                         ((1.0 - one.T.dot(term0))**2)/(one.T.dot(np.dot(R_inv, one))))

    if SSqr <= 1.0e-30:
        if abs(SSqr) <= 1.0e-30:
            NegEI = np.array([-0.0])
        else:
            NegEI = np.array([-0.01])
    else:
        dy = y_min - y_hat
        SSqr = abs(SSqr)
        ei1 = dy*0.5*(1.0 + erf((1/np.sqrt(2))*(dy/np.sqrt(SSqr))))
        ei2 = np.sqrt(SSqr)*(1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*(dy**2/SSqr))
        NegEI = -(ei1 + ei2)

    return NegEI


def init_nodes(N, xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt, xopt, nodeHist, ubd_count):
    pts = (xU_iter-xL_iter) + 1.0
    com_enum = np.prod(pts, axis=0)
    tot_pts = 0.0
    num_cut = min(N-1, com_enum-1)
    if num_cut > 0:
        new_nodes = [[xL_iter, xU_iter, com_enum]]
        for cut in range(num_cut):
            all_area = [item[2] for item in new_nodes]
            maxA = max(all_area)
            ind_maxA = all_area.index(maxA)
            xL_iter, xU_iter, _ = new_nodes[ind_maxA]
            del new_nodes[ind_maxA]

            #Branching scheme stays same
            xloc_iter = np.round(xL_iter + 0.49*(xU_iter - xL_iter))
            # Choose the largest edge
            l_iter = (xU_iter - xL_iter).argmax()
            if xloc_iter[l_iter]<xU_iter[l_iter]:
                delta = 0.5 #0<delta<1
            else:
                delta = -0.5 #-1<delta<0
            for ii in range(2):
                lb = xL_iter.copy()
                ub = xU_iter.copy()
                if ii == 0:
                    ub[l_iter] = np.floor(xloc_iter[l_iter]+delta)
                elif ii == 1:
                    lb[l_iter] = np.ceil(xloc_iter[l_iter]+delta)
                pts = (ub-lb) + 1.0
                enum = np.prod(pts, axis=0)
                new_node = [lb, ub, enum]
                new_nodes.append(new_node)

        args = []
        n_nodes = len(new_nodes)
        for ii in range(n_nodes):
            xL_iter, xU_iter, enum = new_nodes[ii]
            tot_pts += enum
            args.append((xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                         xopt, ii+1, nodeHist, ubd_count))
    else:
        args = [(xL_iter, xU_iter, par_node, LBD_prev, LBD, UBD, fopt,
                xopt, 0, nodeHist, ubd_count)]

    return args

class nodeHistclass():
    def __init__(self):
        self.ubd_track = np.array([1])
        self.ubdloc_best = np.inf
        self.priority_flag = 0

def concave_factor(xI_lb,xI_ub,surrogate):
    xL = (xI_lb - surrogate.X_mean.flatten())/surrogate.X_std.flatten()
    xU = (xI_ub - surrogate.X_mean.flatten())/surrogate.X_std.flatten()
    per_htm = 5.0
    con_fac = np.zeros((len(xL),))
    for k in range(len(xL)):
        if np.abs(xL[k] - xU[k]) > 1.0e-6:
            h_req = (per_htm/100)*(xU[k]-xL[k])
            xm = (xL[k] + xU[k])*0.5
            h_act = (xm-xL[k])*(xm-xU[k])
            con_fac[k] = h_req/h_act
    # print(con_fac)
    return con_fac
