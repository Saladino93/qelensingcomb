import symlens
from symlens import qe

import numpy as np

from orphics import maps, stats

import matplotlib.pyplot as plt

from pixell import enmap, utils as putils

import itertools

import pickle

import re

import pathlib

'''
Summary of useful stuff here:

class Estimator

class LoadfftedMaps

class Converting

class Binner

class Plotting

def fft

def interpolate
'''
#########################################

def Nmodes(L, deltaL, fsky = 1.):
    result = 2*L*deltaL*fsky
    function = interpolation(L, result)
    return function

def Nmodesprecise(lEdges):
    Nmodes = lEdges[1:]**2. - lEdges[:-1]**2
    return Nmodes

#get variance on auto correlation kk, given theory and gaussian noise
def sigma2L(Clkkf, NL):
    return 2*(Clkkf+NL)**2.


def getCov_ij_mn(cim, cjn, cin, cjm, l, deltal, fsky = 0.4, divide_deltal = False):
    result = cim*cjn+cin*cjm
    if divide_deltal:
        result /= (2*deltal*l*fsky)
    return np.nan_to_num(result)

def getcovarianceauto(noises, theorykk, fsky = 1.0):
    
    N = noises.shape[0]
    M = noises.shape[-1]

    theta_ijmn = np.zeros((N, N, N, N, M))

    for i in range(N):
        for j in range(N):
            for m in range(N):
                for n in range(N):
                    data_im = noises[i, m]
                    data_jm = noises[j, m]
                    data_jn = noises[j, n]
                    data_in = noises[i, n]

                    cim = data_im+theorykk
                    cjn = data_jn+theorykk
                    cin = data_in+theorykk
                    cjm = data_jm+theorykk

                    theta_ijmn[i, j, m, n] = getCov_ij_mn(cim = cim, cjn = cjn, cin = cin, cjm = cjm,
                                                l = None, deltal = None, fsky = fsky, divide_deltal = False)
                    
    return theta_ijmn

def getcovariancecross(noises, kk, kg, gg):

    N = noises.shape[0]
    M = noises.shape[-1]
    Q_ij = np.zeros((N, N, M))

    for i in range(N):
        for j in range(N):

            quantity = noises[i, j]

            Q_ij[i, j] = (quantity+kk)*gg+kg**2
            Q_ij[j, i] = (quantity+kk)*gg+kg**2

    return Q_ij

#########################################

def interpolate(l, cl, modlmap):
        return  symlens.interp(l, cl)(modlmap)

def fft(mappa):
    return enmap.samewcs(enmap.fft(mappa, normalize = 'phys'), mappa)

#########################################

def tfm_dict_to_matrix(estimators, dictionary, formato):
    
    keys = list(dictionary.keys())
    N = len(estimators)
    
    all_combs = list(itertools.combinations_with_replacement(list(estimators), 2))
    
    element = dictionary[keys[0]]
    
    Q_ij = np.zeros((N, N, len(element)))
    
    for estA, estB in all_combs:

            try:
                quantity = dictionary[formato(estA, estB)]
            except:
                quantity = dictionary[formato(estB, estA)]

            indexA = estimators.index(estA)
            indexB = estimators.index(estB)

            Q_ij[indexA, indexB] = quantity
            Q_ij[indexB, indexA] = quantity
    
    return Q_ij

def tfm_dict_to_array(estimators, dictionary, formato):
    
    keys = list(dictionary.keys())
    N = len(keys)
    
    element = dictionary[keys[0]]
    
    Q_i = np.zeros((N, len(element)))
    
    for estA in estimators:

            quantity = dictionary[formato(estA)]
            
            indexA = estimators.index(estA)

            Q_i[indexA] = quantity
    
    return Q_i
    
def get_array_from_dict(element, estimators_list = None):
    lista = []
    
    for name in element.keys():
        uppercase = re.sub('[^A-Z]', '', name)
        string = re.sub(r'[A-Z]+-', '', name)
        lista += re.split('-', string)
    if estimators_list is not None:
        estimators = estimators_list
    else:
        estimators = list(set(lista))
        
    if len(estimators) == len(lista):
        formato = lambda estA: f'{uppercase}-{estA}'
        result = tfm_dict_to_array(estimators, element, formato)
    else:
        formato = lambda estA, estB: f'{uppercase}-{estA}-{estB}'
        result = tfm_dict_to_matrix(estimators, element, formato)
        
    return result


def get_element(element, estimators_list = None):
    #If you have a dictionary
    if isinstance(element, dict):  
        result = get_array_from_dict(element, estimators_list)
    #If you do not have a dictionary
    else:
        result = element
    return result

#########################################





def get_mean_and_scatter(N, pmock): #N number of sims, pmock list of cls
    
    mean = np.mean(pmock, axis = 0)
    diff = pmock-mean
     
    scatter = np.sum((diff)**2, axis = 0)/(N-1)
    scatter /= N #for the mean
    scatter = np.sqrt(scatter)

    return mean, scatter


#########################################

class Estimator(object):
    def __init__(self, shape, wcs, feed_dict, estimator,
                 lmin, lmax,
                 field_names = None, groups = None, 
                 Lmin = 20, Lmax = 6000, 
                 hardening = None, estimator_to_harden = 'hu_ok', XY = 'TT'):

        if hardening == '':
            hardening = None        

        xmask, ymask, kmask = self.get_masks(shape, wcs, xlmin = lmin, xlmax = lmax, xlx = None, xly = None,
                                             ylmin = lmin, ylmax = lmax, ylx = None, yly = None,
                                             Lmin = Lmin, Lmax = Lmax, Lx = None, Ly = None)

        self.fdict = feed_dict
           
        f, F, Fr = self.get_mc_expressions(estimator, XY = 'TT', field_names = field_names, estimator_to_harden = estimator_to_harden, 
                           hardening = hardening, feed_dict = feed_dict, shape = shape, wcs = wcs, xmask = xmask, ymask = ymask, kmask = kmask)

        if 'symm' in estimator:
            self.Al = xmask*0.+1.
        else:
            self.Al = self.A_l_custom(shape, wcs, feed_dict, f, F, 
                                  xmask = xmask, ymask = ymask, groups = None, kmask = kmask)
        
        self.F = F
        self.f = f
        self.Fr = Fr

        self.xmask = xmask
        self.ymask = ymask
        self.kmask = kmask
        self.shape, self.wcs = shape, wcs

        self.field_names = field_names

        self.estimator = estimator

        self.XY = XY

    def get_masks(self, shape, wcs, xlmin, xlmax, xlx, xly,
                                    ylmin, ylmax, ylx, yly,
                                    Lmin, Lmax, Lx, Ly):
        xmask = self.get_mask(shape, wcs, xlmin, xlmax, xlx, xly)
        ymask = self.get_mask(shape, wcs, ylmin, ylmax, ylx, yly)
        kmask = self.get_mask(shape, wcs, Lmin, Lmax, Lx, Ly)
        return xmask, ymask, kmask

    def get_mask(self, shape, wcs, lmin, lmax, lx, ly):
        return symlens.mask_kspace(shape, wcs, lxcut = lx, lycut = ly, lmin = lmin, lmax = lmax)

    def A_l_custom(self, shape, wcs, feed_dict, f, F, xmask, ymask, 
                   groups = None,kmask = None):
        return symlens.A_l_custom(shape, wcs, feed_dict, f, F, 
                                  xmask = xmask, ymask = ymask, groups = None,kmask = kmask) 
    
    def reconstruct_other(self, map1, map2, field_names = None, estimator = None, F = None):
        feed_dict = self.fdict.copy()
         
        if field_names is None:
            field_names = self.field_names
            
        name1 = field_names[0]
        name2 = field_names[1]

        feed_dict[name1] = map1
        feed_dict[name2] = map2

        xname1, xname2 = name1+'_l1', name2+'_l2'
        groups = self._get_groups(self.estimator if estimator is None else estimator, noise = False)
        return self._reconstruct(feed_dict, xname = xname1, yname = xname2, F = F)
    
    
    def reconstruct_symm(self, map1, map2):
        recAB = self.reconstruct_other(map1, map2, self.field_names, estimator = 'hdv', F = self.F_phiA)
        reversed_field_names = self.field_names.copy()
        reversed_field_names.reverse()
        recBA = self.reconstruct_other(map2, map1, reversed_field_names, estimator = 'hdv', F = self.F_phiB)
        return self.fdict['wA']*recAB+self.fdict['wB']*recBA
    
    
    def reconstruct(self, map1, map2):
        
        if self.estimator == 'symm':
            mappa = self.reconstruct_symm(map1, map2)
        else:
            mappa = self.reconstruct_other(map1, map2)
        return mappa
        


    def _reconstruct(self,feed_dict,xname ='X_l1',yname = 'Y_l2', groups = None, physical_units = True, F = None):
        uqe = symlens.unnormalized_quadratic_estimator_custom(self.shape,self.wcs,feed_dict,
                                                      self.F if F is None else F,xname = xname,yname = yname,
                                                      xmask = self.xmask,ymask = self.ymask,
                                                      groups = groups,physical_units = physical_units)
        return self.Al * uqe * self.kmask

    def get_Nl_cross(self, Estimator2, tipo = 't'):
        feed_dict = {**self.fdict, **Estimator2.fdict}
        
        if (self.estimator == 'symm') and (Estimator2.estimator == 'symm'):
            N_l_cross_i_j = self.Ncoadd
        
        elif (self.estimator == 'symm' and Estimator2.estimator != 'symm'):
            N_l_cross_i_j = self.get_Nl_cross_symm_with_asymm(feed_dict, self, Estimator2, tipo)   
        
        elif (self.estimator != 'symm' and Estimator2.estimator == 'symm'):
            N_l_cross_i_j = self.get_Nl_cross_symm_with_asymm(feed_dict, Estimator2, self, tipo)
            
        else:
            N_l_cross_i_j= self.get_Nl_cross_other(feed_dict, Estimator2, tipo = tipo)
            
    
        return N_l_cross_i_j
    
    
    def get_Nl_cross_symm_with_asymm(self, feed_dict, EstimatorSymm, EstimatorStd, tipo = 't'):
        
        NLA = self.N_l_cross_custom(EstimatorSymm.shape, EstimatorSymm.wcs, feed_dict, EstimatorSymm.XY, EstimatorStd.XY, EstimatorSymm.F_phiA, EstimatorStd.F, EstimatorStd.Fr,
                                                xmask = EstimatorSymm.xmask*EstimatorStd.xmask, ymask = EstimatorSymm.ymask*EstimatorStd.ymask,
                                                field_names_alpha = EstimatorSymm.field_names, field_names_beta = EstimatorStd.field_names,
                                                falpha = EstimatorSymm.f_phiA, fbeta = EstimatorStd.f, Aalpha = EstimatorSymm.Al, Abeta = EstimatorStd.Al,
                                                groups = self._get_groups('hdv', EstimatorStd.estimator), kmask = EstimatorSymm.kmask*EstimatorStd.kmask,
                                                power_name = tipo)   

        NLB = self.N_l_cross_custom(EstimatorSymm.shape, EstimatorSymm.wcs, feed_dict, EstimatorSymm.XY, EstimatorStd.XY, EstimatorSymm.F_phiB, EstimatorStd.F, EstimatorStd.Fr,
                                                xmask = EstimatorSymm.xmask*EstimatorStd.xmask, ymask = EstimatorSymm.ymask*EstimatorStd.ymask,
                                                field_names_alpha = EstimatorSymm.field_names_r, field_names_beta = EstimatorStd.field_names,
                                                falpha = EstimatorSymm.f_phiB, fbeta = EstimatorStd.f, Aalpha = EstimatorSymm.Al, Abeta = EstimatorStd.Al,
                                                groups = self._get_groups('hdv', EstimatorStd.estimator), kmask = EstimatorSymm.kmask*EstimatorStd.kmask,
                                                power_name = tipo)
        
        result = NLA*EstimatorSymm.fdict['wA']+NLB*EstimatorSymm.fdict['wB']
            
        return result
    
    def get_Nl_cross_other(self, feed_dict, Estimator2, tipo = 't'):
        
        N_l_cross_i_j = self.N_l_cross_custom(self.shape, self.wcs, feed_dict, self.XY, Estimator2.XY, self.F, Estimator2.F, Estimator2.Fr,
                                                    xmask = self.xmask*Estimator2.xmask, ymask = self.ymask*Estimator2.ymask,
                                                    field_names_alpha = self.field_names, field_names_beta = Estimator2.field_names,
                                                    falpha = self.f, fbeta = Estimator2.f, Aalpha = self.Al, Abeta = Estimator2.Al,
                                                    groups = self._get_groups(self.estimator, Estimator2.estimator), kmask = self.kmask*Estimator2.kmask,
                                                    power_name = tipo)

        return N_l_cross_i_j


    def N_l_cross_custom(self, shape, wcs, feed_dict, alpha_XY, beta_XY, Falpha, Fbeta, Fbeta_rev,
                        xmask = None, ymask = None,
                        field_names_alpha = None, field_names_beta = None,
                        falpha = None, fbeta = None, Aalpha = None, Abeta = None,
                        groups = None, kmask = None, power_name = "t"):

        return symlens.N_l_cross_custom(shape, wcs, feed_dict, alpha_XY, beta_XY, Falpha, Fbeta, Fbeta_rev,
                        xmask = xmask, ymask = ymask,
                        field_names_alpha = field_names_alpha, field_names_beta = field_names_beta,
                        falpha = falpha, fbeta = fbeta, Aalpha = Aalpha, Abeta = Abeta,
                        groups = groups, kmask = kmask, power_name = power_name)
    

    def _get_groups(self, Estimator1, Estimator2 = None, noise = True):
        if Estimator2 is None:
            Estimator2 = Estimator1
        return qe._get_groups(Estimator1, Estimator2, noise = noise)
    
    def get_mc_expressions(self, estimator, XY = 'TT', field_names = None, estimator_to_harden = 'hu_ok', 
                           hardening = None, feed_dict = None, shape = None, wcs = None, xmask = None, ymask = None, kmask= None):
        f1, f2 = field_names if field_names is not None else (None,None)
        def t1(ab):
            a,b = ab
            return symlens.e(qe.cross_names(a,b,f1,f1)+"_l1")
        def t2(ab):
            a,b = ab
            return symlens.e(qe.cross_names(a,b,f2,f2)+"_l2")
        X,Y = XY
        if hardening is not None:
            f_phi, F_phi, Fr_phi = self.get_mc_expressions(estimator_to_harden, field_names = field_names, 
					feed_dict = feed_dict, shape = shape, wcs = wcs, xmask = xmask, ymask = ymask, kmask = kmask)
            f_bias, F_bias, _ = self.get_mc_expressions(hardening, field_names = field_names)
            f_bh, F_bh, Fr_bh = self.get_mc_expressions(f'{hardening}-hardened', estimator_to_harden = estimator_to_harden, field_names = field_names, feed_dict = feed_dict, shape = shape, wcs = wcs, xmask = xmask, ymask = ymask, kmask = kmask)
            # 1 / Response of the biasing agent to the biasing agent
            self.fdict[f'A{hardening}_{hardening}_L'] = self.A_l_custom(shape, wcs, feed_dict, f_bias, F_bias,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)
            # 1 / Response of the biasing agent to CMB lensing
            self.fdict[f'Aphi_{hardening}_L'] = self.A_l_custom(shape, wcs, feed_dict, f_phi, F_bias,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)

            f, F, Fr = f_bh, F_bh, Fr_bh
        
        elif 'hardened' in estimator:
            hardening, hardened_name = estimator.split('-')
            assert XY=="TT", "BH only implemented for TT."
            f_phi, F_phi, _ = self.get_mc_expressions(estimator_to_harden, XY, field_names = field_names, feed_dict = feed_dict, shape = shape, wcs = wcs, xmask = xmask, ymask = ymask, kmask = kmask)
            f_src, _, _ = self.get_mc_expressions(hardening, XY, field_names = field_names)
            A_src_src = symlens.e(f'A{hardening}_{hardening}_L')
            A_phi_src = symlens.e(f'Aphi_{hardening}_L')
            f = f_phi - A_src_src / A_phi_src * f_src
            F = f / t1(XY) / t2(XY) / 2
            fr = f
            Fr = F
        elif 'src' in estimator:
            f = symlens.e(f'pc{estimator}_T_T_l1')*symlens.e(f'pc{estimator}_T_T_l2')
            #F = f / t1(XY) / t2(XY) / 2
            F = f / (t1(X+X)*t2(Y+Y)+t1(XY)*t2(XY))
            fr = f
            Fr = F
        elif estimator == 'symm':
            
            f_phiA, F_phiA, Fr_phiA = self.get_mc_expressions('hdv', field_names = field_names)
            
            field_names_r = field_names.copy()
            field_names_r.reverse()
            
            self.field_names_r = field_names_r
            
            f_phiB, F_phiB, Fr_phiB = self.get_mc_expressions('hdv', field_names = field_names_r)
            
            
            self.f_phiA, self.F_phiA, self.Fr_phiA = f_phiA, F_phiA, Fr_phiA
            self.f_phiB, self.F_phiB, self.Fr_phiB = f_phiB, F_phiB, Fr_phiB
    
            AA = self.A_l_custom(shape, wcs, feed_dict, f_phiA, F_phiA,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)
                    
            AB = self.A_l_custom(shape, wcs, feed_dict, f_phiB, F_phiB,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)
                                    
            NA = self.N_l_cross_custom(shape, wcs, feed_dict, XY, XY, F_phiA, F_phiA, Fr_phiA,
                                     xmask = xmask, ymask = ymask, field_names_alpha = field_names, field_names_beta = field_names,
                                     falpha = f_phiA, fbeta = f_phiA,
                                     Aalpha = AA, Abeta = AA, groups = None, kmask = kmask)
            
            NB = self.N_l_cross_custom(shape, wcs, feed_dict, XY, XY, F_phiB, F_phiB, Fr_phiB,
                                     xmask = xmask, ymask = ymask, field_names_alpha = field_names_r, field_names_beta = field_names_r,
                                     falpha = f_phiB, fbeta = f_phiB,
                                     Aalpha = AB, Abeta = AB, groups = None, kmask = kmask)
            
            
            NAB = self.N_l_cross_custom(shape, wcs, feed_dict, XY, XY, F_phiA, F_phiB, Fr_phiB,
                                     xmask = xmask, ymask = ymask, field_names_alpha = field_names, field_names_beta = field_names_r,
                                     falpha = f_phiA, fbeta = f_phiB,
                                     Aalpha = AA, Abeta = AB, groups = None, kmask = kmask)
            
            wA, wB = getasymmweights(NA, NB, NAB)
            
            self.Ncoadd = getcoaddednoise(NA, NB, NAB)
            
            f = f_phiA
            
            F = symlens.e('wA')*F_phiA+symlens.e('wB')*F_phiB
            Fr = symlens.e('wA')*Fr_phiA+symlens.e('wB')*Fr_phiB
            
            self.fdict['wA'] = wA*AA   #NOTE HERE DEFINITION OF WEIGHT
            self.fdict['wB'] = wB*AB
            
        else:
            f_phi, F_phi, Fr_phi = symlens.get_mc_expressions(estimator, XY, field_names, estimator_to_harden)
            f, F, Fr = f_phi, F_phi, Fr_phi

        
        return f, F, Fr
    
    
def getasymmweights(N_E1_E2, N_E2_E1, N_E1_E2_E2_E1):
    w_E1_E2 = N_E2_E1-N_E1_E2_E2_E1
    w_E2_E1 = N_E1_E2-N_E1_E2_E2_E1
    w = N_E1_E2+N_E2_E1-2*N_E1_E2_E2_E1
    w_E1_E2 /= w
    w_E2_E1 /= w
    return np.nan_to_num(w_E1_E2), np.nan_to_num(w_E2_E1)

def getcoaddedmap(map_E1_E2, map_E2_E1, N_E1_E2, N_E2_E1, N_E1_E2_E2_E1):
    w = N_E1_E2+N_E2_E1-2*N_E1_E2_E2_E1
    coadd_noise = (N_E1_E2*N_E2_E1-N_E1_E2_E2_E1**2.)/w
    w_E1_E2, w_E2_E1 = getasymmweights(N_E1_E2, N_E2_E1, N_E1_E2_E2_E1)
    kappatot = np.nan_to_num(w_E1_E2)*map_E1_E2+np.nan_to_num(w_E2_E1)*map_E2_E1
    return kappatot

def getcoaddednoise(N_E1_E2, N_E2_E1, N_E1_E2_E2_E1):
    w = N_E1_E2+N_E2_E1-2*N_E1_E2_E2_E1
    coadd_noise = (N_E1_E2*N_E2_E1-N_E1_E2_E2_E1**2.)/w
    return np.nan_to_num(coadd_noise)

def load_load_spectra(dictionary):
    def load_spectra(A, B):
        return dictionary[A+B]
    return load_spectra

def loadtszprofile(filetxt, modlmap):
    ells, ul = np.loadtxt(filetxt, unpack = True)
    return interpolate(ells, ul, modlmap)

def Loadfeed_dict_function(ells, load_spectra, field_names_A, field_names_B, modlmap, hardeningA = None, hardeningB = None, tszprofileA = None, tszprofileB = None):

    field_names = field_names_A+field_names_B
    all_combs = list(itertools.combinations_with_replacement(list(field_names), 2))

    
    ctt = load_spectra('uCMB', 'uCMB')
    ctt_lensed = load_spectra('lCMB', 'lCMB')
    
    logTT = np.log(ctt)

    theory2dps_lensed_CMB = interpolate(ells, ctt_lensed, modlmap)
    theory2dps_unlensed_CMB = interpolate(ells, ctt, modlmap)
    grad_theory2dps_unlensed_CMB = interpolate(ells, np.gradient(logTT, np.log(ells)), modlmap)
    
    #total2dB = interpolate(ells, fftotB, modlmap)
    #total2dAB = interpolate(ells, fftotAB, modlmap)
    
    feed_dict = {}
    
    feed_dict['uC_T_T'] = theory2dps_unlensed_CMB
    feed_dict['duC_T_T'] = grad_theory2dps_unlensed_CMB

    hardeningA = None if (hardeningA == '') else hardeningA
    hardeningB = None if (hardeningB == '') else hardeningB
 
    if hardeningA is not None:   
        feed_dict[f'pc{hardeningA}_T_T'] = 1. if tszprofileA is None else load_spectra('profile', 'profile')
    if hardeningB is not None:
        feed_dict[f'pc{hardeningB}_T_T'] = 1. if tszprofileB is None else load_spectra('profile', 'profile')

    for A, B in all_combs:
        
        strA = f'_{A}' if A != '' else ''
        strB = f'_{B}' if B != '' else ''
         
        feed_dict[f'tC{strA}_T{strB}_T'] = interpolate(ells, load_spectra(A, B), modlmap)
    
    return feed_dict



    
    
class mapNamesObj():
    def __init__(self, nu):
        self.psmask = lambda x, lmax: f'ps_mask_5mJy_T_patch' 
        self.cmb0template = lambda x: 'cmb0'   
        self.cmb1template = lambda x: 'cmb1'
        self.fgtemplate =  lambda x, lmax: f'sehgal_{x}_large_cutout'
        self.fggausstemplate = lambda x, lmax: f'gaussian_sehgal_{x}_large_cutout' #f'gaussian_sehgal_{x}_{nu}_large_cutout'
        self.kappatemplate = lambda x: 'sehgal_kcmb_large_cutout'
        self.galtemplate = lambda x: f'sehgal_lsstgold_large_cutout'
        self.nu = nu
        
#########################################

def Loadfeed_dict(directory, field_names_A, field_names_B, modlmap, hardeningA = None, hardeningB = None, tszprofileA = None, tszprofileB = None):

    
    ilccase = False
    
    if 'ilc' in field_names_A or 'ilc' in field_names_A[0]:
        ilccase = True

    ell, clunlen, cllen, _, ftot, _, _ = np.loadtxt(directory/'spectra_lensqest_un_len_detectnoise_fftot_fg_ftSZ.txt', unpack = True)
    
    el, ilcpower = np.loadtxt(directory/'power_ilc.txt', unpack = True)

    el, crossilcpower = np.loadtxt(directory/'crosspower_ilc_tszdepr.txt', unpack = True)

    el, deprilcpower = np.loadtxt(directory/'power_ilc_tszdepr.txt', unpack = True)
 
    if ilccase:
        dictionary = {}
        dictionary['ilcilc'] = ilcpower
        dictionary['ilcdeprilcdepr'] = deprilcpower
        dictionary['ilcilcdepr'] = crossilcpower
        dictionary['ilcdeprilc'] = crossilcpower

        dictionary['ilcAilcA'] = ilcpower
        dictionary['ilcAilcB'] = ilcpower
        dictionary['ilcBilcA'] = ilcpower
        dictionary['ilcBilcB'] = ilcpower
        dictionary['ilcilcA'] = ilcpower
        dictionary['ilcilcB'] = ilcpower
        dictionary['ilcAilc'] = ilcpower
        dictionary['ilcBilc'] = ilcpower

        dictionary['ilcdeprilcA'] = crossilcpower
        dictionary['ilcdeprilcB'] = crossilcpower
        dictionary['ilcAilcdepr'] = crossilcpower
        dictionary['ilcBilcdepr'] = crossilcpower

        dictionary['uCMBuCMB'] = np.interp(el, ell, clunlen)
        dictionary['lCMBlCMB'] = np.interp(el, ell, cllen)
    else:
        #field_names_A = field_names_B, and no crosses
        dictionary = {}
        dictionary['uCMBuCMB'] = np.interp(el, ell, clunlen)
        dictionary['lCMBlCMB'] = np.interp(el, ell, cllen)
        for A, B in list(itertools.product(field_names_A, field_names_B)): #zip(field_names_A, field_names_B):
            dictionary[f'{A}{B}'] = np.interp(el, ell, ftot)
    
    filetxt = directory/'tszProfile.txt'
    dictionary['profileprofile'] = loadtszprofile(filetxt, modlmap)
    load_spectra = load_load_spectra(dictionary)
    
    feed_dict = Loadfeed_dict_function(el, load_spectra, field_names_A, field_names_B, modlmap, hardeningA, hardeningB, tszprofileA, tszprofileB)
    
    return feed_dict

#########################################


class LoadfftedMaps():
    def __init__(self, mapsObj, WR, ConvertingObj, changemap, getfft, lmax,
                 tSZ = 'tsz', CIB = 'cib', kSZ = 'ksz', pt = 'radiops', total = 'total'):
        
        self.lmax = lmax

        self.mapsObj = mapsObj

        self.nu = mapsObj.nu
        
        self.tSZ = tSZ
        self.CIB = CIB
        self.kSZ = kSZ
        self.pt = pt
        self.totalFg = total
        
        self.WR = WR
        self.C = ConvertingObj
        
        self.changemap = changemap
        
        self.getfft = getfft
        
    def read_fg(self, fg_name, num, ext = '.txt'):
        
        factor = self.getfgfactor_for_manusmaps(fg_name)
        
        fg_map = self.read(self.mapsObj.fgtemplate(fg_name, self.lmax), num, ext = ext)
        fg_map -= np.mean(fg_map)
        fg_map *= factor
        
        return fg_map
        
    def convert_nu_factors(self, nu_input, nu_output):
        return 0

    
    def read(self, name, i, stringa = '_', ext = '.fits'):
        
        return self.WR.read(name, i, stringa, ext)
   
    def read_shape(self, num = 0):
        cmb0 = self.read(self.mapsObj.cmb0template(''), num, '_', ext = '.txt')
        return cmb0.shape
       

    def read_all(self, fg_name, num):
        '''
        Returns, fg_fft_masked, fg_gaussian_fft_masked, kappa_fft_masked, fft_gal_map
        '''
       
        cmb0 = self.read(self.mapsObj.cmb0template(''), num, '_', ext = '.txt')
        cmb1 = self.read(self.mapsObj.cmb1template(''), num, '_', ext = '.txt')
 
        fg_mask = self.read(self.mapsObj.psmask('', self.lmax), num, '', ext = '.txt')
        
        kappa = self.read(self.mapsObj.kappatemplate(''), num, stringa = '_', ext = '.txt')
        
        gal_map = self.read(self.mapsObj.galtemplate(''), num, ext = '.txt')
       
        cmb0 = self.changemap(cmb0)
        cmb1 = self.changemap(cmb1)    
        kappa = self.changemap(kappa)
        fg_mask = self.changemap(fg_mask)
        gal_map = self.changemap(gal_map)
        
        cmb0_fft = self.getfft(cmb0)
        cmb1_fft = self.getfft(cmb1)
        kappa_fft_masked = self.getfft(kappa)
        fft_gal_map = self.getfft(gal_map)

        fg_fft_masked_1, fg_gaussian_fft_masked_1, fg_fft_masked_2, fg_gaussian_fft_masked_2 = self.read_fg_only(fg_name, num)

        return cmb0_fft, cmb1_fft, fg_fft_masked_1, fg_gaussian_fft_masked_1, fg_fft_masked_2, fg_gaussian_fft_masked_2, kappa_fft_masked, fft_gal_map

        
    def read_fg_only(self, fg_name, num):
        '''
        Returns, fg_fft_masked, fg_gaussian_fft_masked, kappa_fft_masked, fft_gal_map
        '''

        if isinstance(self.nu, list):
            fg_name_1 = fg_name+'_'+self.nu[0]
            fg_name_2 = fg_name+'_'+self.nu[1]
        else:
            fg_name_1 = fg_name+'_'+str(self.nu)
            fg_name_2 = fg_name+'_'+str(self.nu)

        fg_map_1 = self.read_fg(fg_name_1, num)
        fg_map_gauss_1 = self.read(self.mapsObj.fggausstemplate(fg_name_1, self.lmax), num, ext = '.txt')

        fg_map_2 = self.read_fg(fg_name_2, num)
        fg_map_gauss_2 = self.read(self.mapsObj.fggausstemplate(fg_name_2, self.lmax), num, ext = '.txt')

        fg_mask = self.read(self.mapsObj.psmask('', self.lmax), num, '', ext = '.txt')

        fg_map_1 = self.changemap(fg_map_1)
        fg_map_gauss_1 = self.changemap(fg_map_gauss_1)
        fg_map_2 = self.changemap(fg_map_2)
        fg_map_gauss_2 = self.changemap(fg_map_gauss_2)
        fg_mask = self.changemap(fg_mask)

        fg_fft_masked_1 = self.getfft(fg_map_1*fg_mask)
        fg_gaussian_fft_masked_1 = self.getfft(fg_map_gauss_1)
        fg_fft_masked_2 = self.getfft(fg_map_2*fg_mask)
        fg_gaussian_fft_masked_2 = self.getfft(fg_map_gauss_2)

        return fg_fft_masked_1, fg_gaussian_fft_masked_1, fg_fft_masked_2, fg_gaussian_fft_masked_2

    def getfgfactor_for_manusmaps(self, fg_name):
        
        factor = 1.
        
        if isinstance(self.nu, list) or isinstance(self.nu, str):
            return factor
        
        else:
            factor *= 1.e-26
            factor /= self.C.dBdT(self.nu, 2.726)
            factor *= 1.e6

            if self.tSZ in fg_name:
                correction = 0.7
            elif self.CIB in fg_name:
                correction = 0.38
            elif self.kSZ in fg_name:
                correction = 0.82
            elif self.pt in fg_name:
                correction = 1.1
            elif self.totalFg in fg_name:
                correction = 1

            return correction*factor
        
#########################################

class SaveSpectra():
    def __init__(self, directory):
        self.directory = directory 
        

#########################################


class Converting():

    def __init__(self):
        self.c = 3.e8  # m/s
        self.h = 6.63e-34 # SI
        self.kB = 1.38e-23   # SI
        self.Tcmb = 2.726 # K
        #!!! manuwarning
        self.Jansky = 1.e-26 # W/m^2/Hz
        self.Jy = 1.e-26  # [W/m^2/Hz]

    def dBdT(self, nu, T = None):
        '''d(blackbody)/dT, such that
        dI = d(blackbody)/dT * dT
        input: nu [Hz], T thermo temperature of the black body [K]
        output in SI: [W / Hz / m^2 / sr / K]
        '''

        if T is None:
            T = self.Tcmb

        nu *= 1e9

        x = self.h*nu/(self.kB*T)
        result = 2.*self.h**2*nu**4
        result /= self.kB*T**2*self.c**2
        result *= np.exp(x) / (np.exp(x) - 1.)**2
        return result

#########################################


class write_read():
    def __init__(self, directory):
        self.directory = directory
    def set_directory(self, new_directory):
        self.directory = new_directory
    def get_directory(self):
        return self.directory

    def _save(self, directory, name, array, ext = '.fits'):
        enmap.write_map(directory+name+ext, array)
    def save(self, name, i, array, stringa = '_', ext = '.fits'):
        directory = self.get_directory()
        self._save(directory, name+stringa+f'{i}', array, ext)

    def _read(self, directory, name, ext = '.fits'):
        if ext == '.fits':
            return enmap.read_map(directory+name+ext)
        elif ext == '.txt':
            return np.loadtxt(directory+name+ext)
    def read(self, name, i, stringa = '_', ext = '.fits'):
        directory = self.get_directory()
        return self._read(directory, name+stringa+f'{i}', ext)


class dictionary():
    def __init__(self, directory, subdirectory = ''):
        self.directory = pathlib.Path(directory)/subdirectory
        if not self.directory.exists():
            self.directory.mkdir(parents = True, exist_ok = True)
        self.dictionary = {}
    
    def create_subdictionary(self, tag):
        self.add(tag, {})
    
    def add_to_subdictionary(self, tag, tagsub, element):
        self.dictionary[tag][tagsub] = element

    def exists(self, tag):
        return tag in self.dictionary.keys()

    def exists_in_subdictionary(self, tag, tagsub):
        return tagsub in self.dictionary[tag].keys()

    def add(self, tag, element):
        self.dictionary[tag] = element
    def save(self, name):
        name = name+'.pkl'
        data_file = open(self.directory/name,'wb')
        pickle.dump(self.dictionary, data_file)
        data_file.close()

    def read(self, name):
        name = name+'.pkl'
        data_file = open(self.directory/name,'rb')
        output = pickle.load(data_file)
        data_file.close()
        self.dictionary = output
        return self.dictionary
#########################################


class Binner(maps.FourierCalc):
    def __init__(self, shape, wcs, lmin = 10, lmax = 4000, deltal = 10, log = True, nBins = 20):
        '''
        nBins used if log is True
        '''
        super().__init__(shape, wcs)
        self.lmin = lmin
        self.lmax = lmax
        self.deltal = deltal
        self.log = log
        self.nBins = nBins
        
        self.modlmap = self.getmodlmap(shape, wcs)
        
            
    def getmodlmap(self, shape, wcs):
        return enmap.modlmap(shape, wcs)
    
    def get_binner(self, modlmap, bin_edges):
        return symlens.bin2D(modlmap, bin_edges)
        
    def bin_spectra(self, p2d):
        if self.log:
            bin_edges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nBins, 10.)
        else:
            bin_edges = np.arange(self.lmin, self.lmax, self.deltal)
            
        binner = self.get_binner(self.modlmap,bin_edges)
        cents, cl = binner.bin(p2d)
        return cents, cl
    
    def bin_maps(self, map1, map2 = None, pixel_units = False, get_p2d = False):
        '''
        map1, map2 already ffted
        '''
        if map2 is None:
            map2 = map1
        p2d = self.f2power(map1, map2, pixel_units = pixel_units)
        return self.bin_spectra(p2d) if not get_p2d else (0, p2d)
    
    def set_binning(self, lmin, lmax, deltal = None, log = True, nBins = 20):
        self.lmin = lmin
        self.lmax = lmax
        self.deltal = deltal
        self.log = log
        self.nBins = nBins
        if deltal is not None:
            self.deltal = deltal 


#########################################


class Plotting():
    def __init__(self, title, lminplot = None, lmaxplot = None, xscale = 'linear', yscale = 'linear'):
        self.title = title
        self.xscale = xscale
        self.yscale = yscale
        self.lminplot = lminplot
        self.lmaxplot = lmaxplot
        
    def plot_spectra(self, tuples, labels, colors, file_name = None):
        '''
        tuples is basically l, cl
        '''
        for i, multiple in enumerate(zip(tuples, labels, colors)):
            t, label, color = multiple
            l, cl = t
            self.plot(l, cl, label, color)
        plt.legend(loc = 'best')
        if file_name is not None:
            plt.savefig(file_name+'.png')
        plt.title(self.title)
        self.set_scales(self.xscale, self.yscale)
        plt.show()
        
    def plot(self, l, cl, errs = None, label = None, color = None, ls = None):
        if errs is not None:
            plt.fill_between(l, cl-errs, cl+errs, color = color, alpha = 0.4)
            #plt.errorbar(l, cl, errs, label = label, color = color, linestyle = ls)
        plt.plot(l, cl, label = label, color = color, linestyle = ls)
        
    def plotsel(self, l, cl, errs = None, label = None, color = None, ls = None):
        selection = (self.lminplot < l) & (l < self.lmaxplot)
        if errs is not None:
            errs = errs[selection]
        self.plot(l[selection], cl[selection], errs = errs, label = label, color = color, ls = ls)

    def set_scales(self, xscale = None, yscale = None):
        if xscale is None:
            xscale = self.xscale
        if yscale is None:
            yscale = self.yscale
        plt.xscale(xscale)
        plt.yscale(yscale)

#########################################
