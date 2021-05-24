import universe
#reload(universe)
from universe import *

import halo_fit
#reload(halo_fit)
from halo_fit import *

import weight
#reload(weight)
from weight import *

import pn_2d
#reload(pn_2d)
from pn_2d import *

import cmb
#reload(cmb)
from cmb import *

import flat_map
#reload(flat_map)
from flat_map import *


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mock_numb = 80#
delta = mock_numb/size #1 I am using size > mock_numb  #mock_numb/size
start = 0

rank_ass = rank

print('Size', size)
print('Rank', rank)
iMax = (rank_ass+1)*delta+start #number 3 cmbset does not have temperature lensed
iMin = rank_ass*delta+start

#iMin, iMax = 0, 2

iMax = int(iMax)
iMin = int(iMin)

lMin = 30.; lMax = 3.5e3

# ell bins for power spectra
nBins = 21  # number of bins
lRange = (1., 2.*lMax)  # range for power spectra

nu0 = 148

#cmb = StageIVCMB(beam = 1.4, noise = 6., lMin=lMin, lMaxT=lMax, lMaxP=lMax, atm=False)
cmb = CMB(beam=1.4, noise=6., nu1=148.e9, nu2=148.e9, lMin=lMin, lMaxT=lMax, lMaxP=1.e4, fg = True, atm = False, name = "cmbs4")

cmb.nu1 = 148.e9
cmb.nu2 = 148.e9


forCtotal = lambda l: cmb.ftotalTT(l) 
# reinterpolate: gain factor 10 in speed
L = np.logspace(np.log10(lMin/2.), np.log10(2.*lMax), 1001, 10.)
F = np.array(list(map(forCtotal, L)))

L, F = np.loadtxt('ilcresults/power_ilc.txt', unpack = True)

cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

ell = np.arange(1, 10000, 1)

##################################################################################

u = UnivPlanck15()
halofit = Halofit(u, save=False)
w_cmblens = WeightLensSingle(u, z_source=1100., name="cmblens")
p2d_cmblens = P2dAuto(u, halofit, w_cmblens, save=False)
nu = nu0*1.e9


print("My conversion agrees with the Lambda website recommendation:", 1.e-26 / cmb.dBdT(148.e9, cmb.Tcmb), cmb.Tcmb / 1.072480e9)


# cutout dimensions
# map side in lon and lat
dLon = 20.#10.# [deg]
dLat = 20.#10.# [deg]
lonRange = np.array([-dLon/2., dLon/2.]) # [deg]
latRange = np.array([-dLat/2., dLat/2.]) # [deg]
pixRes = 1./60.  #0.5 / 60.  # [arcmin] to [deg]
# number of pixels on the side
xSize = np.int(np.ceil(dLon / pixRes))
ySize = np.int(np.ceil(dLat / pixRes))

baseMap = FlatMap(nX=xSize, nY=ySize, sizeX=dLon*np.pi/180., sizeY=dLat*np.pi/180.)

for iPatch in range(iMin, iMax):
    # calculate the estimators
    print(iPatch)
    direc = '/scratch/r/rbond/omard/CORI17112020/extract_sehgal/manusmaps/flat_maps_large/newmaps11022021/ilc/' 

    grffourier = baseMap.genGRF(cmb.fCtotal, test=False)
    grf = baseMap.inverseFourier(grffourier)
    
    np.savetxt(direc+f'grf_{iPatch}.txt', grf)
