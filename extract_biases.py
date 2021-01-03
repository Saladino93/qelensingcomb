import utilities as u

from mpi4py import MPI

import os, sys

import argparse

import yaml

import itertools

from pixell import enmap, utils as putils

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

Lmin, Lmax = data['Lmin'], data['Lmax']

logmode = data['logmode']
nlogBins = data['nlogBins']
deltal = data['deltalplot']


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

#Prepare for shape, wcs

#Biases calculation


C = u.Converting()


lmin_A, lmax_A, lmin_B, lmax_B = 30, 3500, 30, 3500


for fgnamefile in fgnamefiles:
    for i in range(iMin, iMax):

        dictionary = u.dictionary(savingdirectory)
        
        noisedicttag = 'N'
        trispectrumdicttag = 'T'
        primarydicttag = 'P'
        secondarydicttag = 'S'
        primarycrossdicttag = 'PC'

        dictionary.create_subdictionary(noisedicttag)
        dictionary.create_subdictionary(trispectrumdicttag)
        dictionary.create_subdictionary(primarydicttag)
        dictionary.create_subdictionary(secondarydicttag)
        dictionary.create_subdictionary(primarycrossdicttag)

        for estA, estB in estimatorcombs:
            nuA = estimators_dictionary[estA]['nu']
            nuB = estimators_dictionary[estB]['nu']

            mapsObjA = u.mapNamesObj(nuA)
            mapsObjB = u.mapNamesObj(nuB)

            hardening_A = estimators_dictionary[estA]['hardening']
            hardening_B = estimators_dictionary[estB]['hardening']

            field_names_A = ['A1', 'A2']
            field_names_B = ['B1', 'B2']
                
            if i == iMin:
                changemap = lambda x: enmap.enmap(x, wcs)
                #Load maps for Leg1, Leg2 for estimator A
                LoadA = u.LoadfftedMaps(mapsObj = mapsObjA, WR = WR, ConvertingObj = C, changemap = changemap, getfft = u.fft)
                #Leg1, Leg2, for estimator B
                LoadB = u.LoadfftedMaps(mapsObj = mapsObjB, WR = WR, ConvertingObj = C, changemap = changemap, getfft = u.fft)  
                #Get shape and wcs
                shape = LoadA.read_shape()
                lonCenter, latCenter = 0, 0
                shape, wcs = enmap.geometry(shape = shape, res = 1.*putils.arcmin, pos = (lonCenter, latCenter))
                
                modlmap = enmap.modlmap(shape, wcs)
                #Binner
                Binner = u.Binner(shape, wcs, deltal = deltal, log = logmode, nBins = nlogBins)

                feed_dict = u.Loadfeed_dict(spectra_path, field_names_A, field_names_B, modlmap)

                #Estimator objects
                A = u.Estimator(shape, wcs, feed_dict, estA, lmin_A, lmax_A,
							  field_names = field_names_A, groups = None, Lmin = Lmin, Lmax = Lmax,
							  hardening = hardening_A, XY = 'TT') 

                B = u.Estimator(shape, wcs, feed_dict, estB, lmin_B, lmax_B,
                              field_names = field_names_B, groups = None, Lmin = Lmin, Lmax = Lmax,
                              hardening = hardening_B, XY = 'TT')


                NAB_cross = A.get_Nl_cross(B)
                el, NAB_cross_binned = Binner.bin_spectra(NAB_cross)
                dictionary.add_to_subdictionary(noisedicttag, f'N_{estA}_{estB}', NAB_cross_binned)

            cmb0_fft, cmb1_fft, fg_fft_masked_A1, fg_gaussian_fft_masked_A1, fg_fft_masked_A2, fg_gaussian_fft_masked_A2, kappa_fft_masked, gal_fft_map = LoadA.read_all(fgnamefile, i)			
            if nuA != nuB:
                fg_fft_masked_B1, fg_gaussian_fft_masked_B1, fg_fft_masked_B2, fg_gaussian_fft_masked_B2 = LoadB.read_fg_only(fgnamefile, i)
            else:
                fg_fft_masked_B1, fg_gaussian_fft_masked_B1, fg_fft_masked_B2, fg_gaussian_fft_masked_B2 = fg_fft_masked_A1, fg_gaussian_fft_masked_A1, fg_fft_masked_A2, fg_gaussian_fft_masked_A2



            #Calculate kk
            el, clkk = Binner.bin_maps(kappa_fft_masked, pixel_units = True)
            
            #Calculate kg
            el, clkg = Binner.bin_maps(kappa_fft_masked, gal_fft_map, pixel_units = True)

            #Calculate Q[Tf, Tf], for A and B
            rec_fg_map_A = A.reconstruct(fg_fft_masked_A1, fg_fft_masked_A2)
            rec_fg_gauss_map_A = A.reconstruct(fg_gaussian_fft_masked_A1, fg_gaussian_fft_masked_A2)

            rec_fg_map_B = B.reconstruct(fg_fft_masked_B1, fg_fft_masked_B2)
            rec_fg_gauss_map_B = B.reconstruct(fg_gaussian_fft_masked_B1, fg_gaussian_fft_masked_B2)

            #Calculate trispectrum bias, for A and B 
            el, clfg_A = Binner.bin_maps(rec_fg_map_A, pixel_units = True)
            el, clfg_gauss_A = Binner.bin_maps(rec_fg_gauss_map_A, pixel_units = True)
            el, clfg_B = Binner.bin_maps(rec_fg_map_B, pixel_units = True)
            el, clfg_gauss_B = Binner.bin_maps(rec_fg_gauss_map_B, pixel_units = True)


            el, clfg_A_B = Binner.bin_maps(rec_fg_map_A, rec_fg_map_B)
            el, clfg_gauss_A_B = Binner.bin_maps(rec_fg_gauss_map_A, rec_fg_gauss_map_B)
            trispectrum_A_B = clfg_A_B-clfg_gauss_A_B


            #Calculate primary for auto
            el, primary_A = Binner.bin_maps(kappa_fft_masked, rec_fg_map_A, pixel_units = True)
            el, primary_B = Binner.bin_maps(kappa_fft_masked, rec_fg_map_B, pixel_units = True)
            primary_A_B = primary_A+primary_B

            #Calculate primary for galaxy
            tag_gal = f'P_{estA}'
            if not dictionary.exists_in_subdictionary(primarycrossdicttag, tag_gal):
                el, primary_gal_A = Binner.bin_maps(gal_fft_map, rec_fg_map_A, pixel_units = True)
                dictionary.add_to_subdictionary(primarycrossdicttag, tag_gal, primary_gal_A)

                
            #Calculate secondary for auto
                
            mapS1 = A.reconstruct(cmb0_fft, fg_fft_masked_A2)
            mapS1 = mapS1 + B.reconstruct(fg_fft_masked_B1, cmb0_fft)
            mapS2 = A.reconstruct(cmb1_fft, fg_fft_masked_A2)
            mapS2 = mapS2 + B.reconstruct(fg_fft_masked_A1, cmb1_fft)
                
            el, secondary_A_B = Binner.bin_maps(mapS1, mapS2, pixel_units = True)
            secondary_A_B *= 2

            dictionary.add_to_subdictionary(trispectrumdicttag, f'T_{estA}_{estB}', trispectrum_A_B)
            dictionary.add_to_subdictionary(primarydicttag, f'P_{estA}_{estB}', primary_A_B)
            dictionary.add_to_subdictionary(secondarydicttag, f'S_{estA}_{estB}', secondary_A_B)
      
        dictionary.add('kk', clkk)
        dictionary.add('kg', clkg)
        dictionary.add('ells', el)

        if isinstance(nuA, list):
            nu = nuA[0]
        else:
            nu = nuA

        dictionary.save(f'{fgnamefile}_{nu}_{i}')   

