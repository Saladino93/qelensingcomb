import utilities as u

from mpi4py import MPI

import os, sys

import argparse

import yaml

import itertools

from pixell import enmap, utils as putils

import numpy as np

import pathlib

#Read info

my_parser = argparse.ArgumentParser(description = 'Configuration file.')

my_parser.add_argument('Configuration',
                       metavar='configuration file',
                       type = str,
                       help = 'the path to configuration file')

args = my_parser.parse_args()

values_file = args.Configuration

if not os.path.exists(values_file):
    print('The file specified does not exist')
    sys.exit()

with open(values_file, 'r') as stream:
            data = yaml.safe_load(stream)


Nsims = data['Nsims']


fgnamefiles = data['fgnamefiles']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())
estimatorcombs = list(itertools.combinations_with_replacement(list(estimators), 2))

lista_lmaxes = []

names = {}

for e in estimators:
    elemento = estimators_dictionary[e]
    names[e] = elemento['direc_name']
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]

lmaxes_configs = list(itertools.product(*lista_lmaxes))


lmaxes_configs_input_to_try = data['lmaxes_configs_input_to_try']
if len(lmaxes_configs_input_to_try) > 0:
    lmaxes_configs = lmaxes_configs_input_to_try


Lmin, Lmax = data['Lmin'], data['Lmax']

logmode = data['logmode']
nlogBins = data['nlogBins']
deltal = data['deltalplot']



noisedicttag = data['noisekey']
trispectrumdicttag = data['trispectrumkey']
primarydicttag = data['primarykey']
secondarydicttag = data['secondarykey']
primarycrossdicttag = data['primarycrosskey']

kkkey = data['kkkey']
kgkey = data['kgkey']
ggkey = data['ggkey']
ellskey = data['ellskey']

savingdirectory = data['savingdirectory']
spectra_path = data['spectra_path']
sims_directory = data['sims_directory']
WR = u.write_read(sims_directory)


#MPI configuration

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mock_numb = Nsims#
delta = 1#mock_numb/size #1 #mock_numb/size #1 I am using size > mock_numb  #mock_numb/size
start = 0

print('Size', size)
print('Rank', rank)

rank_ass = rank % mock_numb

iMax = (rank_ass+1)*delta+start
iMin = rank_ass*delta+start

iMax = int(iMax)
iMin = int(iMin)

number_of_groups = size/Nsims
number_of_configs_per_group = int(len(lmaxes_configs)/number_of_groups)
k = int(rank/Nsims)
index = k*number_of_configs_per_group
index_plus = (k+1)*number_of_configs_per_group
lmaxes_configs = lmaxes_configs[index:index_plus]

#Prepare for shape, wcs

#Biases calculation


C = u.Converting()


lmin_A, lmin_B = 30, 30

validationtag = 'val'

#estimatorcombs = [(e, e) for e in estimators]

for fgnamefile in fgnamefiles:
    for lmaxes in lmaxes_configs:
        lmaxes_dict = {}
        lmax_directory = ''
        for e_index, e in enumerate(estimators):
            l = lmaxes[e_index]
            lmaxes_dict[e] = l
            lmax_directory += f'{names[e]}{l}'

        for i in range(iMin, iMax):

            dictionary = u.dictionary(savingdirectory, lmax_directory)
            dictionary.create_subdictionary(noisedicttag)
            dictionary.create_subdictionary(trispectrumdicttag)
            dictionary.create_subdictionary(primarydicttag)
            dictionary.create_subdictionary(secondarydicttag)
            dictionary.create_subdictionary(primarycrossdicttag)
            dictionary.create_subdictionary(validationtag)

            load_nonfg_maps = True

            for estA, estB in estimatorcombs:
                #print('EEEE', estA, estB)
                nuA = estimators_dictionary[estA]['nu']
                nuB = estimators_dictionary[estB]['nu']
                #print(nuA, nuB)
                lmax_A = lmaxes_dict[estA]
                lmax_B = lmaxes_dict[estB]

                mapsObjA = u.mapNamesObj(nuA)
                mapsObjB = u.mapNamesObj(nuB)

                hardening_A = estimators_dictionary[estA]['hardening']
                hardening_B = estimators_dictionary[estB]['hardening']

                #field_names_A = ['A1', 'A2']
                #field_names_B = ['B1', 'B2']
                
                field_names_A = estimators_dictionary[estA]['field_names']
                field_names_B = estimators_dictionary[estB]['field_names']

                
                tszprofileA = estimators_dictionary[estA]['tszprofile']
                tszprofileB = estimators_dictionary[estB]['tszprofile']
                tszprofile_A = None if tszprofileA == '' else 1.
                tszprofile_B = None if tszprofileB == '' else 1.

                changemap = lambda x: enmap.enmap(x, wcs)
                #Load maps for Leg1, Leg2 for estimator A
                LoadA = u.LoadfftedMaps(mapsObj = mapsObjA, WR = WR, ConvertingObj = C, changemap = changemap, getfft = u.fft, lmax = lmax_A)
                #Leg1, Leg2, for estimator B
                LoadB = u.LoadfftedMaps(mapsObj = mapsObjB, WR = WR, ConvertingObj = C, changemap = changemap, getfft = u.fft, lmax = lmax_B) 

                estimator_to_harden_A = 'hu_ok' if (estA in ['bh', 'pbh']) else estA
                estimator_to_harden_B = 'hu_ok' if (estB in ['bh', 'pbh']) else estB
  
                #MAYBE JUST ADD esttohard in yaml config
                
                estimator_to_harden_A = 'symm' if ('symm' in estA) else estimator_to_harden_A #in ['symmbh', 'symmpbh']) else estA
                estimator_to_harden_B = 'symm' if ('symm' in estB) else estimator_to_harden_B #(estB in ['symmbh', 'symmpbh']) else estB
 
                if i == iMin:
                    #Get shape and wcs
                    shape = LoadA.read_shape()
                    lonCenter, latCenter = 0, 0
                    shape, wcs = enmap.geometry(shape = shape, res = 1.*putils.arcmin, pos = (lonCenter, latCenter))
                    modlmap = enmap.modlmap(shape, wcs)
                    #Binner
                    Binner = u.Binner(shape, wcs, lmin = 10, lmax = 4000, deltal = deltal, log = logmode, nBins = nlogBins)

                    feed_dict = u.Loadfeed_dict(pathlib.Path(spectra_path), field_names_A, field_names_B, modlmap, hardening_A, hardening_B, tszprofile_A, tszprofile_B)

                    #NOTE, THIS SHOULD BE OUTSIDE THE IF
                    #BUT IF iMax = iMin+1 , then it should be fine, will make code a bit faster
                    #So this is fine as long as the number of processes is such that the above relation is ok
                    #Estimator objects
                    A = u.Estimator(shape, wcs, feed_dict, estA, lmin_A, lmax_A,
                                    field_names = field_names_A, groups = None, Lmin = Lmin, Lmax = Lmax,
                                    hardening = hardening_A, estimator_to_harden = estimator_to_harden_A, XY = 'TT') 

                    B = u.Estimator(shape, wcs, feed_dict, estB, lmin_B, lmax_B,
                                    field_names = field_names_B, groups = None, Lmin = Lmin, Lmax = Lmax,
                                    hardening = hardening_B, estimator_to_harden = estimator_to_harden_B, XY = 'TT')

                     
                    NAB_cross = A.get_Nl_cross(B)
                    el, NAB_cross_binned = Binner.bin_spectra(NAB_cross)
                    dictionary.add_to_subdictionary(noisedicttag, f'{noisedicttag}-{estA}-{estB}', NAB_cross_binned)


                
                #For now this is necessary only if there are not enough process, so that I can have one process for each i, or iMin-iMax=1
                A = u.Estimator(shape, wcs, feed_dict, estA, lmin_A, lmax_A,
                                    field_names = field_names_A, groups = None, Lmin = Lmin, Lmax = Lmax,
                                    hardening = hardening_A, estimator_to_harden = estimator_to_harden_A, XY = 'TT')

                B = u.Estimator(shape, wcs, feed_dict, estB, lmin_B, lmax_B,
                                    field_names = field_names_B, groups = None, Lmin = Lmin, Lmax = Lmax,
                                    hardening = hardening_B, estimator_to_harden = estimator_to_harden_B, XY = 'TT')
                

                #if you still did not load the maps
                if load_nonfg_maps:
                    cmb0_fft, cmb1_fft, fg_fft_masked_A1, fg_gaussian_fft_masked_A1, fg_fft_masked_A2, fg_gaussian_fft_masked_A2, kappa_fft_masked, gal_fft_map = LoadA.read_all(fgnamefile, i)			
     
                cmb_total = LoadA.read_total_cmb(i) #cmb0_fft+cmb1_fft
                cmb_gauss = LoadA.read_gauss(i) #NOTE THIS IS ONLY WITH ILC POWER                

                fg_fft_masked_A1, fg_gaussian_fft_masked_A1, fg_fft_masked_A2, fg_gaussian_fft_masked_A2 = LoadA.read_fg_only(fgnamefile, i)
                
                if nuA != nuB:
                    fg_fft_masked_B1, fg_gaussian_fft_masked_B1, fg_fft_masked_B2, fg_gaussian_fft_masked_B2 = LoadB.read_fg_only(fgnamefile, i)
                else:
                    fg_fft_masked_B1, fg_gaussian_fft_masked_B1, fg_fft_masked_B2, fg_gaussian_fft_masked_B2 = fg_fft_masked_A1, fg_gaussian_fft_masked_A1, fg_fft_masked_A2, fg_gaussian_fft_masked_A2
                    
                load_nonfg_maps = False

                valtag = f'{validationtag}-{estA}'
                valtagauto = f'{validationtag}-auto-{estA}-{estB}'
                if not dictionary.exists_in_subdictionary(validationtag, valtag) or not dictionary.exists_in_subdictionary(validationtag, valtagauto):
                    rec_cmb_map_A = A.reconstruct(cmb_total, cmb_total)
                    rec_cmb_map_gauss_A = A.reconstruct(cmb_gauss, cmb_gauss)
                    rec_cmb_map_gauss_B = B.reconstruct(cmb_gauss, cmb_gauss)
                    #rec_cmb_map_B = B.reconstruct(cmb_total, cmb_total)
                    el, cross_with_input = Binner.bin_maps(kappa_fft_masked, rec_cmb_map_A, pixel_units = True)
                    dictionary.add_to_subdictionary(validationtag, valtag, cross_with_input) 
                    el, cross_with_auto = Binner.bin_maps(rec_cmb_map_gauss_A, rec_cmb_map_gauss_B, pixel_units = True)
                    dictionary.add_to_subdictionary(validationtag, valtagauto, cross_with_auto)
                    el, cmbpowergauss = Binner.bin_maps(cmb_gauss, cmb_gauss, pixel_units = True)
                    dictionary.add_to_subdictionary(validationtag, 'inputcmbgauss', cmbpowergauss)
            #Calculate kk$
            el, clkk = Binner.bin_maps(kappa_fft_masked, pixel_units = True)
            #Calculate kg$
            el, clkg = Binner.bin_maps(kappa_fft_masked, gal_fft_map, pixel_units = True)
            #Calculate gg$
            el, clgg = Binner.bin_maps(gal_fft_map, gal_fft_map, pixel_units = True)

            dictionary.add(ggkey, clgg)
            dictionary.add(kkkey, clkk)
            dictionary.add(kgkey, clkg)
            dictionary.add(ellskey, el)

            if isinstance(nuA, list):
                nu = nuA[0]
            else:
                nu = nuA

            dictionary.save(f'{fgnamefile}_{nu}_{i}')   

