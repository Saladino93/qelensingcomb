import argparse

import pathlib

import sys

import yaml

import numpy as np

import best

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

plots_directory = data['plotsdirectory']

analysis_directory = data['analysisdirectory']

P = pathlib.Path(analysis_directory)
Pplots = pathlib.Path(plots_directory)

fgnamefiles = data['fgnamefiles']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())


#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']
del estimators_dictionary

noisetag = data['noisekey']
trispectrumtag = data['trispectrumkey']
primarytag = data['primarykey']
secondarytag = data['secondarykey']
primarycrosstag = data['primarycrosskey']

lmin_sel, lmax_sel = data['lmin_sel'], data['lmax_sel']

optversion = data['optversion']

for fgnamefile in fgnamefiles:
    for i in range(1):
   
        fb = 1.

        getoutname = lambda key: f'{key}.npy'
        noises = np.load(P/getoutname(noisetag))
        biases = np.load(P/getoutname('totalbias'))
        kg = np.load(P/getoutname('kg'))
        kk = np.load(P/getoutname('kk'))
        gg = np.load(P/getoutname('gg'))
        ells = np.load(P/getoutname('ells'))
        theta = np.load(P/getoutname('theta'))
        thetacross = np.load(P/getoutname('thetacross'))

        Optimizerkk = best.Opt(estimators, lmin_sel, lmax_sel, ells, kk, theta, biases, noises)
        
        Optimizerkk.optimize(optversion, method = 'diff-ev', gtol = 1000, bounds = [0., 1.], noisebiasconstr = False, fb = fb, inv_variance = True)

        Optimizerkg = best.Opt(estimators, lmin_sel, lmax_sel, ells, kg, thetacross, biases, noises)
        Optimizerkg.optimize(optversion, method = 'diff-ev', gtol = 1000, bounds = [0., 1.], noisebiasconstr = False, fb = fb, inv_variance = True)



