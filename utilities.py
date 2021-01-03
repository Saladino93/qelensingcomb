import symlens
from symlens import qe

import numpy as np

from orphics import maps, stats

import matplotlib.pyplot as plt

from pixell import enmap, utils as putils

import itertools

import pickle

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

def interpolate(l, cl, modlmap):
        return  symlens.interp(l, cl)(modlmap)

def fft(mappa):
    return enmap.samewcs(enmap.fft(mappa, normalize = 'phys'), mappa)



def get_mean_and_cov_matrix(cents, N, pmock): #N number of sims, pmock list of cls
    num_bins = len(cents)
    pmock = pmock.reshape((N, num_bins))
    mean = np.mean(pmock, axis = 0)
    diff = pmock-mean
    m = 0
    for i in range(N):
        cl = diff[i, :]
        m += np.outer(cl, cl)

    m = m/(N-1)

    return mean, m, np.sqrt(np.diag(m))


#########################################

class Estimator(object):
    def __init__(self, shape, wcs, feed_dict, estimator,
                 lmin, lmax,
                 field_names = None, groups = None, 
                 Lmin = 20, Lmax = 6000, 
                 hardening = None, XY = 'TT'):

        if hardening == '':
            hardening = None        

        xmask, ymask, kmask = self.get_masks(shape, wcs, xlmin = lmin, xlmax = lmax, xlx = None, xly = None,
                                             ylmin = lmin, ylmax = lmax, ylx = None, yly = None,
                                             Lmin = Lmin, Lmax = Lmax, Lx = None, Ly = None)

        self.fdict = feed_dict
        
        if hardening is not None:
            f_phi, F_phi, Fr_phi = self.get_mc_expressions('hu_ok', field_names = field_names)
            f_bias, F_bias, _ = self.get_mc_expressions(hardening, field_names = field_names)
            f_bh, F_bh, Fr_bh = self.get_mc_expressions(f'{hardening}-hardened', estimator_to_harden = 'hu_ok', field_names = field_names)
            # 1 / Response of the biasing agent to the biasing agent
            self.fdict[f'A{hardening}_{hardening}_L'] = self.A_l_custom(shape, wcs, feed_dict, f_bias, F_bias,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)
            # 1 / Response of the biasing agent to CMB lensing
            self.fdict[f'Aphi_{hardening}_L'] = self.A_l_custom(shape, wcs, feed_dict, f_phi, F_bias,
                                                        xmask = xmask, ymask = ymask, groups = None, kmask = kmask)

            f, F, Fr = f_bh, F_bh, Fr_bh
        else:
            f_phi, F_phi, Fr_phi = self.get_mc_expressions(estimator, field_names = field_names)
            f, F, Fr = f_phi, F_phi, Fr_phi

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

    def get_Nl(self):
        return self.N_l_cross_custom(self.shape, self.wcs, self.fdict, self.XY, self.XY, self.F, self.F, self.F,
                                     xmask = self.xmask, ymask = self.ymask,
                                     Aalpha = self.Al, Abeta = self.Al, groups = None, kmask = self.kmask)

    def A_l_custom(self, shape, wcs, feed_dict, f, F, xmask, ymask, 
                   groups = None,kmask = None):
        return symlens.A_l_custom(shape, wcs, feed_dict, f, F, 
                                  xmask = xmask, ymask = ymask, groups = None,kmask = kmask)
    
    
    def reconstruct(self, map1, map2):
        feed_dict = self.fdict.copy()
         
        name1 = self.field_names[0]
        name2 = self.field_names[1]

        feed_dict[name1] = map1
        feed_dict[name2] = map2

        xname1, xname2 = name1+'_l1', name2+'_l2'
        groups = self._get_groups(self.estimator, noise = False)
        return self._reconstruct(feed_dict, xname = xname1, yname = xname2)
        


    def _reconstruct(self,feed_dict,xname='X_l1',yname='Y_l2',groups=None,physical_units=True):
        uqe = symlens.unnormalized_quadratic_estimator_custom(self.shape,self.wcs,feed_dict,
                                                      self.F,xname=xname,yname=yname,
                                                      xmask=self.xmask,ymask=self.ymask,
                                                      groups=groups,physical_units=physical_units)
        return self.Al * uqe * self.kmask

    def get_Nl_cross(self, Estimator2, tipo = 't'):
        feed_dict = {**self.fdict, **Estimator2.fdict}
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
    
    def get_mc_expressions(self, estimator, XY = 'TT', field_names = None, estimator_to_harden = 'hu_ok'):
        return symlens.get_mc_expressions(estimator, XY, field_names, estimator_to_harden)


class mapNamesObj():
    def __init__(self, nu):
        self.psmask = lambda x: 'ps_mask_5mJy_T_patch' 
        self.cmb0template = lambda x: 'cmb0'   
        self.cmb1template = lambda x: 'cmb1'
        self.fgtemplate =  lambda x: f'sehgal_{x}_{nu}_large_cutout'
        self.fggausstemplate = lambda x: f'gaussian_sehgal_{x}_{nu}_large_cutout'
        self.kappatemplate = lambda x: 'sehgal_kcmb_large_cutout'
        self.galtemplate = lambda x: f'sehgal_lsstgold_large_cutout'
        self.nu = nu
        
#########################################

def Loadfeed_dict(directory, field_names_A, field_names_B, modlmap):

    field_names = field_names_A+field_names_B
    all_combs = list(itertools.combinations_with_replacement(list(field_names), 2))

    ells, ctt, ctt_lensed, detectnoise, fftot, fg, ftSZ = np.loadtxt(directory, unpack = True)
    logTT = np.log(ctt)

    theory2dps_lensed_CMB = interpolate(ells, ctt_lensed, modlmap)
    theory2dps_unlensed_CMB = interpolate(ells, ctt, modlmap)
    grad_theory2dps_unlensed_CMB = interpolate(ells, np.gradient(logTT, np.log(ells)), modlmap)
    
    #total2dB = interpolate(ells, fftotB, modlmap)
    #total2dAB = interpolate(ells, fftotAB, modlmap)
    total2d = interpolate(ells, fftot, modlmap)
    
    feed_dict = {}
    
    feed_dict['uC_T_T'] = theory2dps_unlensed_CMB
    feed_dict['duC_T_T'] = grad_theory2dps_unlensed_CMB    
    feed_dict['pc_T_T'] = 1.   

    for A, B in all_combs:
        feed_dict[f'tC_{A}_T_{B}_T'] = total2d
    
    return feed_dict

#########################################


class LoadfftedMaps():
    def __init__(self, mapsObj, WR, ConvertingObj, changemap, getfft,
                 tSZ = 'tsz', CIB = 'cib', kSZ = 'ksz', pt = 'radiops', total = 'total'):
        
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
        
        fg_map = self.read(self.mapsObj.fgtemplate(fg_name), num, ext = ext)
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
 
        fg_mask = self.read(self.mapsObj.psmask(''), num, '', ext = '.txt')
        
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
            fg_name_1 = fg_name
            fg_name_2 = fg_name

        fg_map_1 = self.read_fg(fg_name_1, num)
        fg_map_gauss_1 = self.read(self.mapsObj.fggausstemplate(fg_name_1), num, ext = '.txt')

        fg_map_2 = self.read_fg(fg_name_2, num)
        fg_map_gauss_2 = self.read(self.mapsObj.fggausstemplate(fg_name_2), num, ext = '.txt')

        fg_mask = self.read(self.mapsObj.psmask(''), num, '', ext = '.txt')

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
        factor *= 1.e-26
        factor /= self.C.dBdT(self.nu, 2.726)
        factor *= 1.e6

        if fg_name == self.tSZ:
            correction = 0.7
        elif fg_name == self.CIB:
            correction = 0.38
        elif fg_name == self.kSZ:
            correction = 0.82
        elif fg_name == self.pt:
            correction = 1.1
        elif fg_name == self.totalFg:
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
    def __init__(self, directory):
        self.directory = directory
        self.dictionary = {}
    
    def create_subdictionary(self, tag):
        self.add(tag, {})
    
    def add_to_subdictionary(self, tag, tagsub, element):
        self.dictionary[tag][tagsub] = element

    def exists(self, tag):
        return tag in self.dictionary.keys()

    def exists_in_subdictionary(self, tag, tagsub):
        return tag in self.dictionary[tag].keys()

    def add(self, tag, element):
        self.dictionary[tag] = element
    def save(self, name):
        data_file = open(self.directory+name+'.pkl','wb')
        pickle.dump(self.dictionary, data_file)
        data_file.close()

    def read(self, name):
        data_file = open(self.directory+name+'.pkl','rb')
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
    
    def bin_maps(self, map1, map2 = None, pixel_units = False):
        '''
        map1, map2 already ffted
        '''
        if map2 is None:
            map2 = map1
        p2d = self.f2power(map1, map2, pixel_units = pixel_units)
        return self.bin_spectra(p2d)
    
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
        plt.xscale(self.xscale)
        plt.yscale(self.yscale)
        plt.show()
        
    def plot(self, l, cl, label = None, color = None):
        plt.plot(l, cl, label = label, color = color)
        
    def plotsel(self, l, cl, label = None, color = None):
        selection = (self.lminplot < l) & (l < self.lmaxplot)
        self.plot(l[selection], cl[selection], label = label, color = color)

#########################################
