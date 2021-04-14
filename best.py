import numpy as np

import mystic.solvers as my
from mystic.coupler import and_, or_, not_
from mystic.constraints import and_ as _and, or_ as _or, not_ as _not

from mystic.monitors import VerboseMonitor

from mystic.penalty import linear_equality, quadratic_equality
from mystic.constraints import as_constraint

import pathlib

from scipy.ndimage import gaussian_filter1d

class Opt():
    def __init__(self, estimators, lmin_sel, lmax_sel, ells, theory, theta, biases, noises, estimators_to_ignore = None, biases_errors = None):

        self.ells = ells
        self.theta = theta
        self.biases = biases
        self.noises = noises
        self.theory = theory
        self.biases_errors = biases_errors
        #TEMPORARY FOR NOW
        if estimators_to_ignore is not None:
            index = estimators.index(estimators_to_ignore)+1
            self.theta = self.theta[index:, index:, index:, index:, ...]
            self.biases = self.biases[index:, index:, ...]
            self.noises = self.noises[index:, index:, ...]
            estimators = estimators[index:]
            
        
        self.estimators = estimators
        self.lenestimators = len(estimators)
        self.Ne = len(estimators)

        self.lmin_sel = lmin_sel
        self.lmax_sel = lmax_sel

        self._select()

    def filter_(self, x, sigma):
        #temporary
        bin_edges = np.logspace(np.log10(10), np.log10(4000), 15, 10.)
        bin_edges_ = bin_edges[bin_edges>self.lmin_sel]
        bin_edges_ = bin_edges_[bin_edges_<self.lmax_sel]
        deltas = bin_edges_[1:]-bin_edges_[:-1]

        return self.smooth(x, self.ells_selected, deltas, par = sigma)

    def scipy_gaussian(self, x, sigma):
        return gaussian_filter1d(x, sigma)

    def fwhm2sigma(self, fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    def smooth(self, b, ells, deltas, par = 1.):
        sigmas = self.fwhm2sigma(deltas*par)
        smoothed_vals = np.zeros(b.shape)
        Ne = b.shape[0]
        for m in range(Ne):
            for n in range(Ne):
                x = b[m, n]
                for i, pair in enumerate(zip(ells, sigmas)):
                    x_position, sigma = pair
                    kernel = np.exp(-(ells - x_position) ** 2 / (2 * sigma ** 2))
                    kernel = kernel / sum(kernel)
                    smoothed_vals[m, n, i] = sum(x * kernel)
        return smoothed_vals

    def _select(self):
        selected = (self.ells > self.lmin_sel) & (self.ells < self.lmax_sel)
        self.ells_selected = self.ells[selected]

        self.theta_selected = self.theta[..., selected]
        self.biases_selected = self.biases[..., selected]
        self.noises_selected = self.noises[..., selected]

        if self.biases_errors is not None:
            self.biases_errors_selected = self.biases_errors[..., selected] 
        else:
            self.biases_errors_selected = None

        self.theory_selected = self.theory[selected]
        self.nbins = len(self.ells_selected)


    def _get_combined(self, ells, weights_per_l, total, theory):
        z = weights_per_l*total/theory
        biasterm = self.integerate_discrete(z, ells)
        return biasterm

    def get_mv_solution_analytical(self):
        Nrolled = np.nan_to_num(np.rollaxis(self.noises_selected, -1))
        Ne = self.noises_selected.shape[0]
        e = np.ones(Ne)
        weights = np.linalg.inv(Nrolled).dot(e)
        weights /= weights.dot(e.T)[:, None]
        combinedtheta = self.get_variance_part(weights, self.theta_selected)
        weights_l = self.get_mv_weights(self.ells_selected, self.theory_selected, combinedtheta)
        weights = weights.flatten()
        x = np.append(weights, weights_l)
        return weights, weights_l, x

    def get_mv_solution(self, numerical = False, optversion = None):
        if not numerical:
            return self.get_mv_solution_analytical()
        else:
            result = self.optimize(optversion, method = 'diff-ev', gtol = 200, bounds = [0., 1.], noisebiasconstr = False, fb = 0., inv_variance = True, regularise = False)
            a = self.get_a(result.x, True)
            combinedtheta = self.get_variance_part(a, self.theta_selected)
            weights_l = self.get_mv_weights(self.ells_selected, self.theory_selected, combinedtheta)
            weights = result.x
            x = np.append(weights, weights_l)
            return weights, weights_l, x
            

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
       

    def optimize(self, optversion, method = 'diff-ev', gtol = 5000, positive_weights: bool = True, x0 = None, bs0 = None, bounds = [0., 1.], noisebiasconstr = False, fb = 1., inv_variance = False, verbose = True, noiseparameter = 1., regularise = False, threshold = 0.001, regtype = 'std', scale = 0.8, cross = 0.9, npopfactor = 1, ftol = 1e-12, filter_biases = False, sigma = 1.5):
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
            if filter_biases:
                prepare = lambda x: self.filter_(abs(x), sigma = sigma) 
        else:
            prepare = lambda x: x

        f, noisef, biasf = self.get_f_n_b(self.ells_selected, self.theory_selected, self.theta_selected, prepare(self.biases_selected), sum_biases_squared = sum_biases_squared, bias_squared = bias_squared, fb = fb, inv_variance = inv_variance, noiseparameter = noiseparameter)
        self.f = f
        self.noisef = noisef
        self.biasf = biasf

        _, _, biasf_with_sign = self.get_f_n_b(self.ells_selected, self.theory_selected, self.theta_selected, self.biases_selected, sum_biases_squared = sum_biases_squared, bias_squared = bias_squared, fb = fb, inv_variance = inv_variance, noiseparameter = noiseparameter)
        extra_constraint = lambda x: abs(self.noisef(np.array(x))-abs(biasf_with_sign(np.array(x))))

        if self.biases_errors_selected is not None:
            _, _, biasf_error = self.get_f_n_b(self.ells_selected, self.theory_selected, self.theta_selected, self.biases_errors_selected, sum_biases_squared = sum_biases_squared, bias_squared = bias_squared, fb = fb, inv_variance = inv_variance, noiseparameter = noiseparameter)

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


        def get_reg(biases, theorykk, threshold = 0.001, regtype = 'std', lambda_value = 1e-12):
            '''
            regtype: std, extra, biaserror
            '''
            relative = biases/theorykk
            Ne = biases.shape[0]
            
            def loop_over(v):
                selection = np.zeros_like(v, dtype = bool)
                result = 0.
                for i, vx in enumerate(v):
                    selectiontemp = selection
                    selectiontemp[i] = True
                    result += np.sum((1-~selectiontemp*v)**2.)*vx**2.
                return result

            def regel(ai, selection, regtype):
                if regtype == 'std':
                    return np.sum(ai**2*selection)
                elif regtype == 'extra':
                    return regel(ai, selection, regtype = 'std')+loop_over(ai)*lambda_value
            def reg_with_weights(x):
                x = np.array(x)
                a = self.get_a(x, inv_variance).T
                total = 0.
                for i in range(Ne):
                    selection = abs(relative[i, i]) < threshold
                    total += regel(a[i], selection, regtype)
                return total
            
            def regbias(x):
                return biasf_error(x)**2

            if regtype in ['std', 'extra']:
                reg = reg_with_weights
            else:
                reg = regbias
            
            return reg
        
        regulariser = get_reg(self.biases_selected, self.theory_selected, threshold, regtype)

    
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

        if regularise:
            print('Regularizing')
            func = lambda x: f(np.array(x))+regulariser(np.array(x)) 
        else:
            func = lambda x: f(np.array(x))
         
        if method == 'diff-ev': 
            result = my.diffev(func, x0, npop = npopfactor*10*len(list(bnds)), bounds = bnds, ftol = ftol, gtol = gtol, maxiter = 1024**3, maxfun = 1024**3, constraints = constraint_eq, penalty = penalty, full_output = True, itermon = mon, scale = scale, cross = cross)
            #mon = VerboseMonitor(100)
            #result = my.fmin_powell(func, result[0], bounds = bnds, constraints = constraint_eq, penalty = penalty, full_output = True, gtol = 400, maxfun = 1024**3, maxiter = 1024**3, ftol = 1e-7, itermon = mon)
        elif method == 'buckshot':
            result = my.buckshot(lambda x: f(np.array(x)), len(x0), npts = 16, bounds = bnds, constraints = constraint_eq, penalty = penalty, full_output = True, itermon = mon)#, ftol = 1e-6, gtol = gtol, maxiter = 1024**3, maxfun = 1024**3, constraints = constraint_eq, penalty = penalty, full_output=True, itermon = mon)
        elif method == 'powell':
            result = my.fmin_powell(func, x0, bounds = bnds, constraints = constraint_eq, penalty = penalty, full_output = True, gtol = gtol, maxfun = 1024**3, maxiter = 1024**3, ftol = 1e-7, itermon = mon)
        elif method == 'lattice':
            result = my.lattice(func, ndim = len(x0), nbins = 100, bounds = bnds, ftol = 1e-10, gtol = gtol, maxiter = 1024**3, maxfun = 1024**3, constraints = constraint_eq, penalty = penalty, full_output = True, itermon = mon)
        else:
            print(f'{method} not implemented!')
        
        history = np.hstack((np.array(mon.x), np.array(mon.y)[:, None]))
        
        result = Res(result[0], self.ells_selected, history)
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
    def __init__(self, risultato = None, ells = None, history = None):
        self.x = risultato
        self.set_ells(ells)
        self.history = history
    
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
    def save_history(self, path, name, hname = 'h_'):
        nome = hname+name
        if self.history is not None:
            np.save(path/nome, self.history)
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
        self.save_history(path, name)

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
