import numpy as np

from scipy.interpolate import interp1d

import sys
sys.path.append('LensQuEst-1/')

import cmb as cmbmod
import foregrounds_utils as cmbmod

from flat_map import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mock_numb = 80#
delta = mock_numb/size #1 #mock_numb/size #1 I am using size > mock_numb  #mock_numb/size
start = 0
iMax = (rank+1)*delta+start #number 3 cmbset does not have temperature lensed
iMin = rank*delta+start

iMax = int(iMax)
iMin = int(iMin)


tszname = 'tsz'
cibname = 'cib'
kszname = 'ksz'
radiopsname = 'radiops'
totalname = 'total'

outname = tszname

rank0 = -1

if outname == totalname:
    fgnames = [tszname, cibname, kszname, radiopsname]
else:
    fgnames = [outname]

ell, nl27, nl39, nl93, nl145, nl225, nl280, nl27x39, nl93x145, nl225x280 = np.loadtxt('../ASO_LAT_Nell_T_baseline_fsky0p4.txt', unpack = True)

nfreqs = 6

nus = np.array([27, 39, 93, 145, 225, 280])*1e9

lMin = 10.
lMaxT = 1.e4
lMaxP = 1.e4

cmb = np.empty((nfreqs, nfreqs), dtype = object)

npower = np.zeros((nfreqs, nfreqs, len(ell)))
npower[0, 0] = nl27
npower[1, 1] = nl39
npower[2, 2] = nl93
npower[3, 3] = nl145
npower[4, 4] = nl225
npower[5, 5] = nl280
npower[0, 1] = nl27x39
npower[1, 0] = npower[0, 1]
npower[2, 3] = nl93x145
npower[3, 2] = npower[2, 3]
npower[4, 5] = nl225x280
npower[5, 4] = npower[4, 5]

covariance = np.zeros((len(ell), nfreqs, nfreqs))
covariance_fg_only = np.zeros((len(ell), nfreqs, nfreqs))

AtSZ = np.zeros(nfreqs)
AkSZ = np.zeros(nfreqs)
ACIB = np.zeros(nfreqs)
Aradiops = np.zeros(nfreqs)



nu0 = 148e9

cmbnu0 = cmbmod.Foregrounds(nu0, nu0)
#cmbnu0 = cmbmod.CMB(beam = 1., noise = 1., nu1 = nu0, nu2 = nu0, lMin = lMin, lMaxT = lMaxT, lMaxP = lMaxP, fg = True, atm = False)

for i in range(nfreqs):
         for j in range(nfreqs):
            noise = 0.
            beam = 1.
            noisepower = interp1d(ell, npower[i, j], kind = 'linear', bounds_error = False, fill_value = 0.) 
            cmb[i,j] = cmbmod.Foregrounds(nus[i], nus[j])
            #cmb[i,j] = cmbmod.CMB(beam = beam, noise = noise, nu1 = nus[i], nu2 = nus[j], lMin = lMin, lMaxT = lMaxT, lMaxP = lMaxP, fg = True, atm = False)
            cmb[i,j].ftotalTT = lambda l: cmb[i,j].flensedTT(l) + cmb[i,j].fkSZ(l) + cmb[i,j].fCIB(l) + cmb[i,j].ftSZ(l) + cmb[i,j].ftSZ_CIB(l) + cmb[i,j].fradioPoisson(l) + noisepower(l)
            covariance[:, i, j] = cmb[i,j].ftotalTT(ell)
            if outname == totalname:
                fgonly = lambda l: cmb[i,j].fkSZ(l) + cmb[i,j].fCIB(l) + cmb[i,j].ftSZ(l) + cmb[i,j].ftSZ_CIB(l) + cmb[i,j].fradioPoisson(l)
            elif outname == tszname:
                fgonly = lambda l: cmb[i,j].ftSZ(l) 
            elif outname == cibname:
                fgonly = lambda l: cmb[i,j].fCIB(l)
            elif outname == kszname:
                fgonly = lambda l: cmb[i,j].fkSZ(l) 
            elif outname == radiopsname:
                fgonly = lambda l: cmb[i,j].fradioPoisson(l)
            covariance_fg_only[:, i, j] = fgonly(ell)
            AtSZ[i] = cmb[i,j].tszFreqDpdceTemp(nus[i])
            AkSZ[i] = cmb[i,j].kszFreqDpdceTemp(nus[i])*0.+1.
            ACIB[i] = cmb[i,j].cibPoissonFreqDpdceTemp(nus[i])
            Aradiops[i] = cmb[i,j].radioPoissonFreqDpdceTemp(nus[i])


if outname == totalname:
    fgonly148 = lambda l: cmbnu0.fkSZ(l) + cmbnu0.fCIB(l) + cmbnu0.ftSZ(l) + cmbnu0.ftSZ_CIB(l) + cmbnu0.fradioPoisson(l)
elif outname == cibname:
    fgonly148 = lambda l: cmbnu0.fCIB(l)
elif outname == kszname:
    fgonly148 = lambda l: cmbnu0.fkSZ(l)
elif outname == tszname:
    fgonly148 = lambda l: cmbnu0.ftSZ(l)
elif outname == radiopsname:
    fgonly148 = lambda l: cmbnu0.fradioPoisson(l)


if rank == rank0:
    np.savetxt('ilcresults/tSZpowers.txt', np.c_[ell, cmbnu0.ftSZ(ell), cmb[0, 0].ftSZ(ell), cmb[1, 1].ftSZ(ell), cmb[2, 2].ftSZ(ell), cmb[3, 3].ftSZ(ell), cmb[4, 4].ftSZ(ell), cmb[5, 5].ftSZ(ell)])
    np.savetxt(f'ilcresults/{outname}power148.txt', np.c_[ell, fgonly148(ell)])

e = np.ones(nfreqs)

A = np.c_[e, AtSZ]

Cinv = np.linalg.inv(covariance)
Cinv_e = np.einsum('...ij, ...j->...i', Cinv, e)
e_Cinv_e = np.einsum('...i, ...i -> ...', e, Cinv_e)

w = Cinv_e*(e_Cinv_e[:, np.newaxis])**-1.
w = np.nan_to_num(w)
wilc_calculated = w.copy()

power = np.einsum('...i, ...ij, ...j->...', w, covariance, w)

power_fg_only = np.einsum('...i, ...ij, ...j->...', w, covariance_fg_only, w)

if rank == rank0:
    np.savetxt('ilcresults/w_ilc.txt', np.c_[ell, w])
    np.save('ilcresults/covariance', covariance)
    np.savetxt('ilcresults/power_ilc.txt', np.c_[ell, power])
    np.savetxt(f'ilcresults/power_fg_only_ilc_{outname}.txt', np.c_[ell, power_fg_only])

e = np.array([1, 0])

Cinv_A = np.einsum('...ij, jk->...ik', Cinv, A)
A_Cinv_A = np.einsum('ik, ...il -> ...kl', A, Cinv_A)
inv_A_Cinv_A = np.linalg.inv(A_Cinv_A)

prod = np.einsum('...il, ...lk -> ...ik', Cinv_A, inv_A_Cinv_A)

w = np.einsum('...ij, j', prod, e)
w = np.nan_to_num(w)

power = np.einsum('...i, ...ij, ...j->...', w, covariance, w)
crosspower = np.einsum('...i, ...ij, ...j->...', w, covariance, wilc_calculated)

power_fg_only = np.einsum('...i, ...ij, ...j->...', w, covariance_fg_only, w)

if rank == rank0:
    np.savetxt('ilcresults/w_ilc_'+str(outname)+'depr.txt', np.c_[ell, w])
    np.savetxt('ilcresults/power_ilc_'+str(outname)+'depr.txt', np.c_[ell, power])
    np.savetxt('ilcresults/crosspower_ilc_'+str(outname)+'depr.txt', np.c_[ell, crosspower])
    np.savetxt('ilcresults/power_fg_only_ilc_'+str(outname)+'depr.txt', np.c_[ell, power_fg_only])
file_ = np.loadtxt('ilcresults/w_ilc.txt')
ell, wilc = file_[:, 0], file_[:, 1:]
file_ =  np.loadtxt('ilcresults/w_ilc_'+str(outname)+'depr.txt')
ell, wilc_depr = file_[:, 0], file_[:, 1:]

dLon = 20.# [deg]
dLat = 20.# [deg]
lonRange = np.array([-dLon/2., dLon/2.]) # [deg]
latRange = np.array([-dLat/2., dLat/2.]) # [deg]
pixRes = 1./60.  #0.5 / 60.  # [arcmin] to [deg]
# number of pixels on the side
xSize = np.int(np.ceil(dLon / pixRes))
ySize = np.int(np.ceil(dLat / pixRes))

baseMap = FlatMap(nX=xSize, nY=ySize, sizeX=dLon*np.pi/180., sizeY=dLat*np.pi/180.)

product_ilc_tSZ = np.einsum('...i, i -> ...', wilc, AtSZ)
product_ilc_depr_tSZ = np.einsum('...i, i -> ...', wilc_depr, AtSZ)

product_ilc_kSZ = np.einsum('...i, i -> ...', wilc, AkSZ)
product_ilc_depr_kSZ = np.einsum('...i, i -> ...', wilc_depr, AkSZ)

product_ilc_CIB = np.einsum('...i, i -> ...', wilc, ACIB)
product_ilc_depr_CIB = np.einsum('...i, i -> ...', wilc_depr, ACIB)

product_ilc_radiops = np.einsum('...i, i -> ...', wilc, Aradiops)
product_ilc_depr_radiops = np.einsum('...i, i -> ...', wilc_depr, Aradiops)

interpolated_ilc = interp1d(ell, product_ilc_tSZ, kind = 'linear', bounds_error = False, fill_value = 0.)
interpolated_ilc_depr = interp1d(ell, product_ilc_depr_tSZ, kind = 'linear', bounds_error = False, fill_value = 0.)
W_ilc_tSZ = interpolated_ilc(baseMap.l)
W_ilc_depr_tSZ = interpolated_ilc_depr(baseMap.l)

interpolated_ilc = interp1d(ell, product_ilc_kSZ, kind = 'linear', bounds_error = False, fill_value = 0.)
interpolated_ilc_depr = interp1d(ell, product_ilc_depr_kSZ, kind = 'linear', bounds_error = False, fill_value = 0.)
W_ilc_kSZ = interpolated_ilc(baseMap.l)
W_ilc_depr_kSZ = interpolated_ilc_depr(baseMap.l)

interpolated_ilc = interp1d(ell, product_ilc_CIB, kind = 'linear', bounds_error = False, fill_value = 0.)
interpolated_ilc_depr = interp1d(ell, product_ilc_depr_CIB, kind = 'linear', bounds_error = False, fill_value = 0.)
W_ilc_CIB = interpolated_ilc(baseMap.l)
W_ilc_depr_CIB = interpolated_ilc_depr(baseMap.l)

interpolated_ilc = interp1d(ell, product_ilc_radiops, kind = 'linear', bounds_error = False, fill_value = 0.)
interpolated_ilc_depr = interp1d(ell, product_ilc_depr_radiops, kind = 'linear', bounds_error = False, fill_value = 0.)
W_ilc_radiops = interpolated_ilc(baseMap.l)
W_ilc_depr_radiops = interpolated_ilc_depr(baseMap.l)


nu0 = 148*1.e9
fnu0tsz = cmb[0, 0].tszFreqDpdceTemp(nu0)
fnu0ksz = cmb[0, 0].kszFreqDpdceTemp(nu0)*0.+1.
fnu0CIB = cmb[0, 0].cibPoissonFreqDpdceTemp(nu0)
fnu0radiops = cmb[0, 0].radioPoissonFreqDpdceTemp(nu0)

cmb_ = cmb[0, 0]
cmb_.nu1 = nu0
#iMin, iMax = 0, 80

lMaxes = [4500]

for lMax in lMaxes:
    for iPatch in range(iMin, iMax):
        print('Do patch', iPatch)
        pp = "../manusmaps/"
        #path = pp+"flat_maps_large/sehgal_"+str(fgname)+"_148/sehgal_tsz_148_large_cutout_"+str(iPatch)+".txt"
        total = 0.
        total_depr = 0.
        total_148 = 0.
        for fgname in fgnames:
            path = pp+'flat_maps_large/newmaps11022021/148/'+"sehgal_"+str(fgname)+"_148_large_cutout_"+str(iPatch)+".txt"  
            tsz = np.genfromtxt(path)

            if fgname == tszname:
                factor_fg = 0.7
                W_ilc = W_ilc_tSZ
                W_ilc_depr = W_ilc_depr_tSZ
                fnu0 = fnu0tsz
            elif fgname == cibname:
                fnu0 = fnu0CIB
                factor_fg = 0.38
                W_ilc = W_ilc_CIB
                W_ilc_depr = W_ilc_depr_CIB
            elif fgname == kszname:
                fnu0 = fnu0ksz
                factor_fg = 0.82
                W_ilc = W_ilc_kSZ
                W_ilc_depr = W_ilc_depr_kSZ
            elif fgname == radiopsname:
                fnu0 = fnu0radiops
                factor_fg = 1.1
                W_ilc = W_ilc_radiops
                W_ilc_depr = W_ilc_depr_radiops
            elif fgname == totalname:
                factor_fg = 1.

            tsz = factor_fg*tsz
        
            tsz -= np.mean(tsz.flatten())
            conversion = 1.e6 * 1.e-26 / cmb_.dBdT(cmb_.nu1, cmb_.Tcmb)
            tsz *= conversion

            tsz /= fnu0
            fluxCut = 0.005
            maskTot = np.loadtxt(pp+'flat_maps_large/newmaps11022021/148/'+'ps_mask_'+str(np.int(round(fluxCut*1000)))+"mJy_T_patch"+str(iPatch)+".txt")

            tsz *= maskTot

            tszFourier = baseMap.fourier(tsz)  

            nBins = 21
            if outname == totalname:
                total_148 += tszFourier*fnu0
                if fgname == radiopsname: # the last name on the list
                    lCen, Cl, sCl = baseMap.powerSpectrum(total_148, plot=False, save=False, nBins=nBins)
                    np.savetxt(f'ilcresults/ilcspectra/p_148_{outname}_lmax_{lMax}'+str(iPatch)+'.txt', np.c_[lCen, Cl, fgonly148(lCen)])
            else:
                lCen, Cl, sCl = baseMap.powerSpectrum(tszFourier*fnu0, plot=False, save=False, nBins=nBins)
                np.savetxt(f'ilcresults/ilcspectra/p_148_{outname}_lmax_{lMax}'+str(iPatch)+'.txt', np.c_[lCen, Cl, fgonly148(lCen)])


            tszFourier_ilc = tszFourier*W_ilc 
            tszFourier_ilc_depr = tszFourier*W_ilc_depr

            total += tszFourier_ilc
            total_depr += tszFourier_ilc_depr

        tszFourier_ilc = total
        tszFourier_ilc_depr = total_depr

        f = lambda lx,ly: np.exp(1j*np.random.uniform(0., 2.*np.pi))
        tszGFourier_ilc = baseMap.filterFourier(f, dataFourier = tszFourier_ilc)
        tszG_ilc = baseMap.inverseFourier(tszGFourier_ilc)
        tszGFourier_ilc_depr = baseMap.filterFourier(f, dataFourier = tszFourier_ilc_depr)
        tszG_ilc_depr = baseMap.inverseFourier(tszGFourier_ilc_depr)    

        tsz_ilc = baseMap.inverseFourier(tszFourier_ilc)
        tsz_ilc_depr = baseMap.inverseFourier(tszFourier_ilc_depr)
        fgname = outname
        pathilc = pp+"flat_maps_large/newmaps11022021/ilc/sehgal_"+str(fgname)+"_ilc_large_cutout_"+str(iPatch)+".txt"
        pathilc_depr = pp+"flat_maps_large/newmaps11022021/ilc/sehgal_"+str(fgname)+"_ilc_depr_large_cutout_"+str(iPatch)+".txt"
        pathilcG = pp+"flat_maps_large/newmaps11022021/ilc/gaussian_sehgal_"+str(fgname)+"_ilc_large_cutout_"+str(iPatch)+".txt"
        pathilcG_depr = pp+"flat_maps_large/newmaps11022021/ilc/gaussian_sehgal_"+str(fgname)+"_ilc_depr_large_cutout_"+str(iPatch)+".txt"
        
        np.savetxt(pathilc, tsz_ilc)
        np.savetxt(pathilc_depr, tsz_ilc_depr)
        np.savetxt(pathilcG, tszG_ilc)
        np.savetxt(pathilcG_depr, tszG_ilc_depr)
        nBins = 21
        lCen, Cl, sCl = baseMap.powerSpectrum(tszFourier_ilc, plot=False, save=False, nBins=nBins)
        lCen, Cldepr, sCl = baseMap.powerSpectrum(tszFourier_ilc_depr, plot=False, save=False, nBins=nBins)

        np.savetxt(f'ilcresults/ilcspectra/p{outname}_'+str(iPatch)+'.txt', np.c_[lCen, Cl, Cldepr])
