"""
Genetic algorithm used to assist Branch and Bound.
"""
import copy

from six.moves import range

import numpy as np

from pyDOE import lhs


class Genetic_Algorithm():
    """
    This is the Simple Genetic Algorithm implementation
    based on 2009 AAE550: MDO Lecture notes of
    Prof. William A. Crossley
    """
    def __init__(self, objfun):
        self.objfun = objfun

        self.lchrom = 0
        self.npop = 0
        self.elite = True

    def execute_ga(self, vlb, vub, bits=8, pop_size=24, max_gen=1000):
        xopt = copy.deepcopy(vlb)
        fopt = np.inf
        self.lchrom = int(np.sum(bits))

        if np.mod(pop_size, 2) == 1:
            pop_size += 1
        self.npop = int(pop_size)
        fitness = np.zeros((self.npop, ))

        Pc = 0.5
        Pm = (self.lchrom + 1.0) / (2.0 * pop_size * np.sum(bits))
        elite = self.elite

        # new_gen = 1 - np.round(np.random.rand(self.npop, self.lchrom))
        new_gen = np.round(lhs(self.lchrom, self.npop, criterion='center'))
        # new_gen, lchrom = encode(x0, vlb, vub, bits) #TODO: from an user-supplied intial population

        # Main Loop
        nfit = 0
        for generation in range(max_gen + 1):
            old_gen = copy.deepcopy(new_gen)
            x_pop  = self.decode(old_gen, vlb, vub, bits)

            # Evaluate points in this generation.
            for ii in range(self.npop):
                x = x_pop[ii]
                fitness[ii] = self.objfun(x)
                nfit += 1

            # Elitism means replace worst performing point with best from previous generation.
            if elite and generation > 0:
                max_index = np.argmax(fitness)
                old_gen[max_index] = min_gen
                x_pop[max_index] = min_x
                fitness[max_index] = min_fit

            # Find best performing point in this generation.
            min_fit = np.min(fitness)
            min_index = np.argmin(fitness)
            min_gen = old_gen[min_index]
            min_x = x_pop[min_index]

            if min_fit < fopt:
                fopt = min_fit
                xopt = min_x

            # Evolve new generation.
            new_gen = self.tournament(old_gen, fitness)
            new_gen = self.crossover(new_gen, Pc)
            new_gen = self.mutate(new_gen, Pm)

        return xopt, fopt, nfit

    def tournament(self, old_gen, fitness):
        i_old_gen = np.array(range(self.npop))
        new_gen = []
        for j in range(2):
            old_gen, i_shuffled = self.shuffle(old_gen)
            fitness = fitness[i_shuffled]
            i_old_gen = i_old_gen[i_shuffled]
            index = np.array(range(0, self.npop - 1, 2))
            i_min = np.zeros((len(index), 1))
            for ii in range(len(index)):
                i_min[ii] = np.argmin(np.array([fitness[index[ii]], fitness[index[ii] + 1]]))
            selected = i_min.flatten() + range(0, self.npop - 1, 2)
            for ii in range(len(selected)):
                if j == 0 and ii == 0:
                    new_gen = np.array([old_gen[int(selected[ii])]]) #np.concatenate((new_gen,old_gen[selected[ii]]),axis = 0)
                else:
                    new_gen = np.append(new_gen, np.array([old_gen[int(selected[ii])]]), axis=0)
        return new_gen

    def crossover(self, old_gen, Pc):
        new_gen = copy.deepcopy(old_gen)
        sites = np.random.rand(self.npop//2, self.lchrom)
        for i in range(self.npop//2):
            for j in range(self.lchrom):
                if sites[i][j] < Pc:
                    new_gen[2 * i][j] = old_gen[2 * i + 1][j]
                    new_gen[2 * i + 1][j] = old_gen[2 * i][j]
        return new_gen

    def mutate(self, old_gen, Pm):
        temp = np.random.rand(self.npop, self.lchrom)
        new_gen = copy.deepcopy(old_gen)
        for ii in range(self.npop):
            for jj in range(self.lchrom):
                if temp[ii][jj] < Pm:
                    new_gen[ii][jj] = 1 - old_gen[ii][jj]
        return new_gen

    def shuffle(self, old_gen):
        """
        Shuffle (reorder) the points in the population.

        Used in tournament selection.

        Parameters
        ----------
        old_gen : ndarray
            Old population.

        Returns
        -------
        ndarray
            New shuffled population.
        ndarray(dtype=np.int)
            Index array that maps the shuffle from old to new.
        """
        temp = np.random.rand(self.npop)
        index = np.argsort(temp)
        return old_gen[index], index

    def decode(self, gen, vlb, vub, bits):
        no_para = len(bits)
        coarse = (vub - vlb)/(2**bits - 1)
        x = np.zeros((self.npop, no_para))
        for kk in range(self.npop):
            sbit = 1
            ebit = 0
            for jj in range(no_para):
                ebit = int(bits[jj]) + ebit
                accum = 0.0
                ADD = 1
                for ii in range(sbit, ebit + 1):
                    pbit = ii + 1 - sbit
                    if gen[kk][ii - 1] == 1:
                        if ADD == 1:
                            accum = accum + (2.0**(bits[jj] - pbit + 1) - 1)
                            ADD = 0
                        else:
                            accum = accum - (2.0**(bits[jj] - pbit + 1) - 1)
                            ADD = 1
                x[kk][jj] = accum * coarse[jj] + vlb[jj]
                sbit = ebit + 1
        return x

    def encode(self, x, vlb, vub, bits):
        return

    def test_func(self, x):
        ''' Solution: xopt = [0.2857, -0.8571], fopt = 23.2933'''
        A = (2*x[0] - 3*x[1])**2;
        B = 18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2;
        C = (x[0] + x[1] + 1)**2;
        D = 19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2;

        f = (30 + A*B)*(1 + C*D);
        return f
