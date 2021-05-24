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

fgnames = [tszname]

dLon = 20.# [deg]
dLat = 20.# [deg]
lonRange = np.array([-dLon/2., dLon/2.]) # [deg]
latRange = np.array([-dLat/2., dLat/2.]) # [deg]
pixRes = 1./60.  #0.5 / 60.  # [arcmin] to [deg]
# number of pixels on the side
xSize = np.int(np.ceil(dLon / pixRes))
ySize = np.int(np.ceil(dLat / pixRes))

baseMap = FlatMap(nX=xSize, nY=ySize, sizeX=dLon*np.pi/180., sizeY=dLat*np.pi/180.)

lMaxes = [4500]


nu0 = 148e9

cmb_ = cmbmod.Foregrounds(nu0, nu0)

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

            tsz = factor_fg*tsz
        
            tsz -= np.mean(tsz.flatten())
            conversion = 1.e6 * 1.e-26 / cmb_.dBdT(cmb_.nu1, cmb_.Tcmb)
            tsz *= conversion

            fluxCut = 0.005
            maskTot = np.loadtxt(pp+'flat_maps_large/newmaps11022021/148/'+'ps_mask_'+str(np.int(round(fluxCut*1000)))+"mJy_T_patch"+str(iPatch)+".txt")

            tsz *= maskTot

            tszFourier = baseMap.fourier(tsz)  
            nBins = 42
            lCen, Cl, sCl = baseMap.powerSpectrum(tszFourier, plot=False, save=False, nBins=nBins)
            np.savetxt(f'ilcresults/ilcspectra/p_148_{outname}_'+str(iPatch)+'.txt', np.c_[lCen, Cl])


