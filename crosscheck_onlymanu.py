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

lMin = 30.; lMax = 4.5e3

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
cmb.fCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)

ell = np.arange(1, 10000, 1)


if rank == 0:
    ffunl = np.vectorize(lambda l: cmb.funlensedTT(l))
    fflen = np.vectorize(lambda l: cmb.flensedTT(l))
    ffdetectornoise = np.vectorize(lambda l: cmb.fdetectorNoise(l))
    fftot = np.vectorize(forCtotal)
    ffg = np.vectorize(lambda l: cmb.fkSZ(l) + cmb.fCIB(l) + cmb.ftSZ(l) + cmb.ftSZ_CIB(l) + cmb.fradioPoisson(l))
    fftsz = np.vectorize(lambda l: cmb.ftSZ(l))
    np.savetxt('spectra_lensqest_un_len_detectnoise_fftot_fg_ftSZ.txt', np.c_[ell, ffunl(ell), fflen(ell), ffdetectornoise(ell), fftot(ell), ffg(ell), fftsz(ell)])
    np.savetxt('all_fgs_tsz_cib_cross_ksz_radio.txt', np.c_[ell, np.vectorize(lambda l: cmb.ftSZ(l))(ell), np.vectorize(lambda l: cmb.fCIB(l))(ell), np.vectorize(lambda l: cmb.ftSZ_CIB(l))(ell), np.vectorize(lambda l: cmb.fkSZ(l))(ell), np.vectorize(lambda l: cmb.fradioPoisson(l))(ell)])

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

if rank == 0:
    fNqCmb_fft = baseMap.forecastN0Kappa(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, test=False)
    np.savetxt('QEnoise.txt', np.c_[ell, fNqCmb_fft(ell), p2d_cmblens.fPinterp(ell)])


for iPatch in range(iMin, iMax):
    pp = "../manusmaps/"
    # read Sehgal maps [Jy/sr]
    path = pp+"flat_maps_large/newmaps11022021/148/sehgal_cib_148_large_cutout_"+str(iPatch)+".txt"
    cib148 = np.genfromtxt(path)
    path = pp+"flat_maps_large/newmaps11022021/148/sehgal_tsz_148_large_cutout_"+str(iPatch)+".txt"
    #path = pp+"flat_maps_large/latestmaps13092020/sehgal_tsz_148_large_cutout_"+str(iPatch)+".txt"
    tsz = np.genfromtxt(path)
    path = pp+"flat_maps_large/newmaps11022021/148/sehgal_ksz_148_large_cutout_"+str(iPatch)+".txt"
    ksz = np.genfromtxt(path)
    path = pp+"flat_maps_large/newmaps11022021/148/sehgal_radiops_148_large_cutout_"+str(iPatch)+".txt"
    radiops = np.genfromtxt(path)

    # Rescale the maps to match Dunkley power spectra after masking
    cib148 *= 0.38 # 0.35 * np.sqrt(1.2)
    tsz *= 0.7  #0.82 # 0.68 * np.sqrt(1.45)
    ksz *= 0.82 # 0.8 * np.sqrt(1.05)
    radiops *= 1.1

    # Sum of all the foregrounds
    fTot = cib148 + tsz + ksz + radiops
    #np.savetxt(pp+'flat_maps_large/latestmaps13092020/'+'sehgal_total_148_large_cutout_'+str(iPatch)+".txt", fTot)
    # Fourier transform


    tsz = tsz

    tszFourier = baseMap.fourier(tsz)
    fTotFourier = baseMap.fourier(fTot) 

    # Create a mask with the sum of the foregrounds as the input
    fluxCut = 0.005  # in Jy 
    maskPatchRadius = 3. * np.pi/(180.*60.)   # in rad
    #cmb.ftotal
    #forCtotal = lambda l: cmb.flensedTT(l)+cmb.fkSZ(l) + cmb.fCIB(l) + cmb.ftSZ(l) + cmb.ftSZ_CIB(l) + cmb.fradioPoisson(l)#cmb.ftotalTT(l)
    #F = np.array(list(map(forCtotal, L)))
    #forCtotal = interp1d(L, F, kind='linear', bounds_error=False, fill_value=0.)
    profile = None #cmb.fbeam 
    maskTot = baseMap.pointSourceMaskMatchedFilterIsotropic(cmb.fCtotal, fluxCut, fprof=profile, dataFourier=fTotFourier, maskPatchRadius=maskPatchRadius, test=False)
    np.savetxt(f"prova{iPatch}.txt", maskTot)
    #np.savetxt(pp+'flat_maps_large/latestmaps13092020/'+'ps_mask_'+str(np.int(round(fluxCut*1000)))+"mJy_T_patch"+str(iPatch)+".txt", maskTot)
    #maskTot = np.loadtxt(pp+'flat_maps_large/latestmaps13092020/'+'ps_mask_'+str(np.int(round(fluxCut*1000)))+"mJy_T_patch"+str(iPatch)+".txt")
    #maskTot = 0.*maskTot+1.
    tsz -= np.mean(tsz.flatten())

    # convert from Jy/sr to muK
    # consistent with Sehgal's conversion: https://lambda.gsfc.nasa.gov/toolbox/tb_sim_info.cfm
    conversion = 1.e6 * 1.e-26 / cmb.dBdT(cmb.nu1, cmb.Tcmb) 
    tsz *= conversion

    # mask all the maps 
    tsz *= maskTot

    # Fourier transform
    tszFourier = baseMap.fourier(tsz)

    # randomize the phases to create "Gaussian" version of tSZ map
    f = lambda lx,ly: np.exp(1j*np.random.uniform(0., 2.*np.pi))
    tszGFourier = baseMap.filterFourier(f, dataFourier=tszFourier)
    tszG = baseMap.inverseFourier(tszGFourier)
    tszG *= maskTot

    
    path = pp+"flat_maps_large/latestmaps13092020/gaussian_sehgal_tsz_148_large_cutout_"+str(iPatch)+".txt"
    #np.savetxt(path, tszG)
    tszGFourier = baseMap.fourier(tszG)
    
    QE = 'QE'
    SH = 'SH'

    estimator = QE

    # calculate the estimators

    if estimator == QE:
        pQFourier = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszFourier, test=False)
        pQGFourier = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszGFourier, test=False)
    elif estimator == SH:
        pQFourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszFourier, test=False)
        pQGFourier = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszGFourier, test=False)

    #pQFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszFourier, test=False)
    #pQGFourier = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.flensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier=tszGFourier, test=False)

    # get the power spectra
    lCen, ClQ, sCl = baseMap.powerSpectrum(pQFourier, plot=False, save=False, nBins=nBins)         
    lCen, ClQG, sCl = baseMap.powerSpectrum(pQGFourier, plot=False, save=False, nBins=nBins)

    # this is the trispectrum bias
    ClQ -= ClQG

    #i = iPatch
    #cmb0 = np.loadtxt('/global/cscratch1/sd/omard/extract_sehgal/manusmaps/flat_maps_large/cmb0/cmb0_'+str(iPatch)+'.txt')
    #cmb1 = np.loadtxt('/global/cscratch1/sd/omard/extract_sehgal/manusmaps/flat_maps_large/cmb1/cmb1_'+str(iPatch)+'.txt')

    #cmb0 = np.loadtxt(pp+'flat_maps_large/latestmaps13092020/cmb0_'+str(iPatch)+'.txt')#*maskTot
    #cmb1 = np.loadtxt(pp+'flat_maps_large/latestmaps13092020/cmb1_'+str(iPatch)+'.txt')#*maskTot

    #cmb0Fourier = baseMap.fourier(cmb0)
    cmb0Fourier = baseMap.genGRF(cmb.funlensedTT) #baseMap.fourier(cmb0)
    cmb0 = baseMap.inverseFourier(cmb0Fourier)
    manusdir = pp+'flat_maps_large/'
    kCmb = np.genfromtxt(manusdir+'sehgal_kcmb/sehgal_kcmb_large_cutout_'+str(iPatch)+'.txt')#*maskTot
    kCmbFourier = baseMap.fourier(kCmb)
    cmb1 = baseMap.doLensingTaylor(cmb0, kappaFourier=kCmbFourier, order=1) - cmb0
    cmb1Fourier = baseMap.fourier(cmb1)

    #np.savetxt(pp+'flat_maps_large/latestmaps13092020/cmb0_'+str(iPatch)+'.txt', cmb0)
    #np.savetxt(pp+'flat_maps_large/latestmaps13092020/cmb1_'+str(iPatch)+'.txt', cmb1)

    totalCmbFourier = tszFourier

    if estimator == QE:
        qCmbFourier0f = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb0Fourier, dataFourier2 = totalCmbFourier, test=False)
        qCmbFourierf0 = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb0Fourier, test=False)
        qCmbFourier1f = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb1Fourier, dataFourier2 = totalCmbFourier, test=False)
        qCmbFourierf1 = baseMap.computeQuadEstKappaNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb1Fourier, test=False)
    elif estimator == SH:
        qCmbFourier0f = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb0Fourier, dataFourier2 = totalCmbFourier, test=False)
        qCmbFourierf0 = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb0Fourier, test=False)
        qCmbFourier1f = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb1Fourier, dataFourier2 = totalCmbFourier, test=False)
        qCmbFourierf1 = baseMap.computeQuadEstKappaShearNormCorr(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb1Fourier, test=False)

    #qCmbFourier0f = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb0Fourier, dataFourier2 = totalCmbFourier, test=False)
    #qCmbFourierf0 = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb0Fourier, test=False)
    #qCmbFourier1f = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = cmb1Fourier, dataFourier2 = totalCmbFourier, test=False)
    #qCmbFourierf1 = baseMap.computeQuadEstKappaPointSourceHardenedNorm(cmb.funlensedTT, cmb.fCtotal, lMin=lMin, lMax=lMax, dataFourier = totalCmbFourier, dataFourier2 = cmb1Fourier, test=False)

    

    lCen, Clcrossf0f1, sCl = baseMap.crossPowerSpectrum(qCmbFourierf0, qCmbFourierf1, plot=False, save=False, nBins=nBins)
    
    lCen, Clcross0f1f, sCl = baseMap.crossPowerSpectrum(qCmbFourier0f, qCmbFourier1f, plot=False, save=False, nBins=nBins)

    lCen, Clcross0ff1, sCl = baseMap.crossPowerSpectrum(qCmbFourier0f, qCmbFourierf1, plot=False, save=False, nBins=nBins)

    lCen, Clcross1ff0, sCl = baseMap.crossPowerSpectrum(qCmbFourier1f, qCmbFourierf0, plot=False, save=False, nBins=nBins)

    lCen, clkk, sCl = baseMap.crossPowerSpectrum(kCmbFourier, kCmbFourier, plot=False, save=False, nBins=nBins)

    #np.savetxt('out_txt/'+estimator+'_sec_0f1f_f0f1_0ff1_1ff0.txt'+str(iPatch)+'.txt', np.c_[Clcross0f1f, Clcrossf0f1, Clcross0ff1, Clcross1ff0])

    #Clcross = (2*Clcross0f1f)+(2*Clcrossf0f1)+2*(Clcross0ff1+Clcross1ff0)

    lCen, Clcross, sCl = baseMap.crossPowerSpectrum((qCmbFourier0f+qCmbFourierf0), (qCmbFourier1f+qCmbFourierf1), plot=False, save=False, nBins=nBins) 
    Clcross *= 2

    lCen, Clprim, sCL = baseMap.crossPowerSpectrum(pQFourier, kCmbFourier, plot=False, save=False, nBins=nBins)
    Clprim *= 2.


    np.savetxt('out_txt/'+estimator+'_'+str(lMax)+'_tris_prim_sec_'+str(iPatch)+'.txt', np.c_[lCen, ClQ, Clprim, Clcross, clkk, p2d_cmblens.fPinterp(lCen)])


