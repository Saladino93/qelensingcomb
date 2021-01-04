import argparse

import pathlib

import sys

import utilities as u

import yaml

import numpy as np

#Read info

my_parser = argparse.ArgumentParser(description = 'Configuration file.')

my_parser.add_argument('Configuration',
                       metavar='configuration file',
                       type = str,
                       help = 'the path to configuration file')

args = my_parser.parse_args()

values_file = args.Configuration

if not pathlib.Path(values_file).exists():
    print('The file specified does not exist')
    sys.exit()

with open(values_file, 'r') as stream:
            data = yaml.safe_load(stream)

directory = data['savingdirectory']

plots_directory = data['plotsdirectory']

analysis_directory = data['analysisdirectory']

P = pathlib.Path(analysis_directory)
if not P.exists():
    P.mkdir(parents = True, exist_ok = True)

Pplots = pathlib.Path(plots_directory)
if not Pplots.exists():
    Pplots.mkdir(parents = True, exist_ok = True)

fgnamefiles = data['fgnamefiles']

Nsims = data['Nsims']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())
#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']

del estimators_dictionary

for fgnamefile in fgnamefiles:
    for i in range(1):

        dic = u.dictionary(directory)
    
        i = 0 
        dictionary_temp = dic.read(f'{fgnamefile}_{nu}_{i}')
    
        #Loop over the elements of the saved dictionary
        for k in dictionary_temp.keys():
        
            total = []
        
            for j in range(Nsims):
                dictionary = dic.read(f'{fgnamefile}_{nu}_{j}')
                array = u.get_element(dictionary[k])
                total += [array]
        
            total = np.array(total)
        
            mean, scatter = u.get_mean_and_scatter(Nsims, total)
                        
            outname = f'{k}.npy'
            np.save(P/outname, mean)
        
            els = ['ells']
            if not k in els:
                outname = f'scatter_{k}.npy'
                #NOTE, correction for the old Manus and Simos sims
                A, A_octanct = 81*20**2, 5156.6
                factor = np.sqrt(A/A_octanct)
                np.save(P/outname, scatter)


Plotting = u.Plotting('Biases', lminplot = 30, lmaxplot = 2000, xscale = 'log')
