import numpy as np

import mystic.solvers as my
from mystic.coupler import and_, or_, not_
from mystic.constraints import and_ as _and, or_ as _or, not_ as _not

from mystic.monitors import VerboseMonitor

from mystic.penalty import linear_equality, quadratic_equality
from mystic.constraints import as_constraint

import pathlib

class Opt():
    def __init__(self, estimators, lmin_sel, lmax_sel, ells, theory, theta, biases, noises, estimators_to_ignore = None):

        self.ells = ells
        self.theta = theta
        self.biases = biases
        self.noises = noises
        self.theory = theory

        #TEMPORARY FOR NOW
        if estimators_to_ignore is not None:
            index = estimators.index(estimators_to_ignore)+1
            self.theta = self.theta[index:, index:, index:, index:, ...]
            self.biases = self.biases[index:, index:, ...]
            self.noises = self.noises[index:, index:, ...]
            estimators = estimators[index:]
            
            
        self.estimators = estimators
        self.lenestimators = len(estimators)

        self.lmin_sel = lmin_sel
        self.lmax_sel = lmax_sel

        self._select()

    def _select(self):
        selected = (self.ells > self.lmin_sel) & (self.ells < self.lmax_sel)
        self.ells_selected = self.ells[selected]

        self.theta_selected = self.theta[..., selected]
        self.biases_selected = self.biases[..., selected]
        self.noises_selected = self.noises[..., selected]

        self.theory_selected = self.theory[selected]
        self.nbins = len(self.ells_selected)


    def _get_combined(self, ells, weights_per_l, total, theory):
        z = weights_per_l*total/theory
        biasterm = self.integerate_discrete(z, ells)
        return biasterm


    def get_mv_weights(self, ells, theory, variance):
        weights = theory**2/variance
        norm = self.integerate_discrete(weights, ells)
        weights = weights/norm
        return weights


    def get_variance_part(self, a, theta):
        if theta.ndim > 3:
            #this is for auto
            variance_part = np.einsum('...i, ...j, ...k, ...m , ijkm...->...', a, a, a, a, theta)
        else:
            variance_part = np.einsum('...i, ...j, ij...->...', a, a, theta)
        return variance_part

    def get_final_variance_weights(self, x, ells, theory, theta, inv_variance):
        a = self.get_a(x, inv_variance)
        variance = self.get_variance_part(a, theta)
        return self.get_weight_per_l(x, ells, theory, variance, inv_variance)

    def get_bias_part(self, a, bias):
        if bias.ndim > 2:
            #this is for auto
            bias_part = np.einsum('...i, ...j, ij...->...', a, a, bias)
        else:
            bias_part = np.einsum('...i, i...->...', a, bias)
        return bias_part

    def get_per_l_part(self, a, matrix):
        if matrix.ndim > 2:
            #this is for auto
            part = np.einsum('...i, ...j, ij...->...', a, a, matrix)
        return part


    def get_bias_term(self, ells, theory, bias, a, weight_per_l):
        bias_part = self.get_bias_part(a, bias)
        biasterm = self._get_combined(ells, weight_per_l, bias_part, theory)
        return biasterm


    def get_a(self, x, inv_variance):
        a = x[:-self.nbins].reshape(self.nbins, self.lenestimators) if not inv_variance else x.reshape(self.nbins, self.lenestimators)
        return a

    def get_weight_per_l(self, x, ells, theory, variance_part, inv_variance):
        if inv_variance:
            weight_per_l = self.get_mv_weights(ells, theory, variance_part)
        else:
            weight_per_l = x[-self.nbins:]
        return weight_per_l

    def get_f_n_b(self, ells, theory, theta, bias, sum_biases_squared = False, bias_squared = False, fb = 1., inv_variance = False, noiseparameter = 1.):
        
        def f(x):
            a = self.get_a(x, inv_variance)
            variance_part = self.get_variance_part(a, theta)

            weight_per_l = self.get_weight_per_l(x, ells, theory, variance_part, inv_variance)

            total_result = 0.

            biasterm = self.get_bias_term(ells, theory, bias, a, weight_per_l)**2.
            squarednoiseterm = self._get_combined(ells, weight_per_l**2., variance_part, theory**2.) 

            total_result = noiseparameter*squarednoiseterm+biasterm*fb

            return total_result


        def noisef(x):
            a = self.get_a(x, inv_variance)
            variance_part = self.get_variance_part(a, theta)
            weight_per_l = self.get_weight_per_l(x, ells, theory, variance_part, inv_variance)
            squarednoiseterm = self._get_combined(ells, weight_per_l**2., variance_part, theory**2.)
            noiseterm = np.sqrt(squarednoiseterm)
            return noiseterm

        def biasf(x):
            a = self.get_a(x, inv_variance)
            variance_part = self.get_variance_part(a, theta)
            weight_per_l = self.get_weight_per_l(x, ells, theory, variance_part, inv_variance)
            biasterm = self.get_bias_term(ells, theory, bias, a, weight_per_l)
            return biasterm

        return f, noisef, biasf

    def integerate_discrete(self, y, ells):
        #Nmodes = lEdges[1:]**2. - lEdges[:-1]**2
        factor = 4*np.pi
        return np.trapz(y*ells, ells)*(2*np.pi)/(2*np.pi)**2*factor
       

    def optimize(self, optversion, method = 'diff-ev', gtol = 5000, positive_weights: bool = True, x0 = None, bs0 = None, bounds = [0., 1.], noisebiasconstr = False, fb = 1., inv_variance = False, verbose = True, noiseparameter = 1.):
        '''
        Methods: diff-ev, SLSQP
        '''
        
        if verbose:
            print(f'Start optimization with {method}')

            if inv_variance:
                print('Using combined inverse variance weights')

        if x0 is None:
            x0 = []
            if self.lenestimators == 2:
                v = np.random.rand(1)/2
            elif self.lenestimators == 3:
                v = np.random.rand(2)/3
            elif self.lenestimators == 4:
                v = np.random.rand(3)/4
            for a in v:
                x0 += [a]
            x0 += [1.-np.sum(v)]
            x0 = np.array(x0*int(self.nbins))

        if bs0 is None:
            bs0 = np.ones(int(self.nbins))
            norma = self.integerate_discrete(bs0, self.ells_selected)
            bs0 /= norma
        
        dims = (self.lenestimators+1) if not inv_variance else self.lenestimators
        bnds = [(bounds[0], bounds[1]) for i in range(dims*self.nbins)]
        bnds = tuple(bnds)

        if inv_variance:
            x0 = x0
        else:
            x0 = np.append(x0, bs0)

        #if positive_weights:
        #    cons = ({'type': 'eq', 'fun': self.get_constraint()}, {'type': 'ineq', 'fun': self.get_constraint_ineq()})
        #else:
        #    cons = ({'type': 'eq', 'fun': self.get_constraint()})


        weights_name = optversion['weights_name']

        if verbose:
            print(f'Doing for {weights_name}')

        sum_biases_squared = optversion['sum_biases_squared']
        abs_biases = optversion['abs_biases']
        bias_squared = optversion['bias_squared']

        if abs_biases:
            prepare = lambda x: abs(x)
        else:
            prepare = lambda x: x

        f, noisef, biasf = self.get_f_n_b(self.ells_selected, self.theory_selected, self.theta_selected, prepare(self.biases_selected), sum_biases_squared = sum_biases_squared, bias_squared = bias_squared, fb = fb, inv_variance = inv_variance, noiseparameter = noiseparameter)
        self.f = f
        self.noisef = noisef
        self.biasf = biasf

        extra_constraint = lambda x: abs(self.noisef(np.array(x))-self.biasf(np.array(x)))


        def constraint_eq(x):
            x = np.array(x)
            a = self.get_a(x, inv_variance)
            a[:, -1] = 1-np.sum(a[:, :-1], axis = 1)
            if not inv_variance:
                x[:-self.nbins] = a.flatten()
            else:
                x = a.flatten()
            return x

        def penalty1(x):
            x = np.array(x)
            b = x[-self.nbins:]
            res = self.integerate_discrete(b, self.ells_selected)
            return 1-res

        k = 1e20

        if noisebiasconstr:
            gtol = gtol
            @quadratic_equality(condition=penalty1, k = k)
            @quadratic_equality(condition=extra_constraint, k = k)
            def penalty(x):
                return 0.0
        else:
            gtol = gtol
            @quadratic_equality(condition=penalty1, k = k)
            def penalty(x):
                return 0.0       

        if inv_variance:
            penalty = None 
            if noisebiasconstr:
                @quadratic_equality(condition=extra_constraint, k = k)
                def penalty(x):
                    return 0.0

        mon = VerboseMonitor(100)

        func = lambda x: f(np.array(x))
        result = my.diffev(func, x0, npop = 10*len(list(bnds)), bounds = bnds, ftol = 1e-11, gtol = gtol, maxiter = 1024**3, maxfun = 1024**3, constraints = constraint_eq, penalty = penalty, full_output=True, itermon=mon)

        result = Res(result[0], self.ells_selected)
        self.result = result
        
        ws = self.get_weights(result.x, inv_variance, verbose = verbose)        
        weights_per_l = self.get_final_variance_weights(result.x, self.ells_selected, self.theory_selected, self.theta_selected, inv_variance)

        result.set_weights(tuple(list(ws)+[weights_per_l]))
        
        self.monitor = mon
        
        return result
       
    def get_weights(self, x, inv_variance, verbose = True):

        aa = self.get_a(x, inv_variance)
        if verbose:
            print('Weights in columns', aa)
            print('Sum', np.sum(aa, axis = 1))

        ns = aa.shape[1]
        lista = []
        for i in range(ns):
            lista += [aa[:, i]]
        ws = tuple(lista)
        return ws

 
class Res():
    def __init__(self, risultato = None, ells = None):
        self.x = risultato
        self.set_ells(ells)
    
    def set_ells(self, ells):
        self.ells = ells
        if ells is not None:
            self.nbins = len(ells)

    def set_weights(self, ws):
        self.ws = ws

    def save(self, element, path, name):
        nome = name
        P = self._create_path(path)
        np.save(path/nome, element)

    def load(self, path, name):
        nome = name+'.npy'
        P = self._create_path(path)
        return np.load(path/nome)

    def save_x(self, path, name, xname = 'x_'):
        nome = xname+name
        np.save(path/nome, self.x)
    def save_weights(self, path, name, wname = 'w_'):
        nome = wname+name
        np.save(path/nome, np.c_[[self.ells]+list(self.ws)])
    def load_x(self, path, name, xname = 'x_'):
        nome = xname+name+'.npy'
        self.x = np.load(path/nome)
    def load_weights(self, path, name, wname = 'w_'):
        nome = wname+name+'.npy'
        f = np.load(path/nome).T
        self.set_ells(f[:, 0])
        self.ws = f[:, 1:]

    def _create_path(self, path):
        P = pathlib.Path(path)
        if not P.exists():
            P.mkdir(parents = True, exist_ok = True)
        return P
    def save_all(self, path, name):
        P = self._create_path(path)
        self.save_x(P, name)
        self.save_weights(path, name)
    def load_all(self, path, name):
        P = self._create_path(path)
        self.load_x(P, name)
        self.load_weights(path, name)
        self.wx = self.x[:-self.nbins]
    def plot(self, path, name):
        return 0

class ResultsPlotter():
    def __init__(self):
        return 0
