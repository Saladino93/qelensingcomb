import argparse

import pathlib

import sys

import yaml

import numpy as np

import best

import itertools

my_parser = argparse.ArgumentParser(description = 'Configuration file.')

my_parser.add_argument('Configuration',
                       metavar='configuration file',
                       type = str,
                       help = 'the path to configuration file')

my_parser.add_argument('fb',
                       metavar='bias enhancement',
                       type = float)

my_parser.add_argument('gtol',
                       metavar='gtol for diff-ev',
                       type = int)

my_parser.add_argument('noisebiasconstr',
                       metavar='if noise=bias constraint',
                       type = int, 
                       help = '0 for False, 1 for True')

my_parser.add_argument('invvariance',
                       metavar='inverse variance weights',
                       type = int,
                       help = '0 for False, 1 for True')

my_parser.add_argument('h',
                       metavar='huok',
                       type = int)

my_parser.add_argument('s',
                       metavar='shear',
                       type = int)

my_parser.add_argument('b',
                       metavar='biashard',
                       type = int)


args = my_parser.parse_args()

values_file = args.Configuration
fb = args.fb
gtol = args.gtol
noisebiasconstr = bool(args.noisebiasconstr)
invvariance = bool(args.invvariance)
h = args.h
s = args.s
b = args.b


if not pathlib.Path(values_file).exists():
    print('The file specified does not exist')
    sys.exit()

with open(values_file, 'r') as stream:
            data = yaml.safe_load(stream)

plots_directory = data['plotsdirectory']

analysis_directory = data['analysisdirectory']

results_directory = data['resultsdirectory']

PP = pathlib.Path(analysis_directory)
Pplots = pathlib.Path(plots_directory)

fgnamefiles = data['fgnamefiles']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())


lista_lmaxes = []

names = {}
for e in estimators:
    elemento = estimators_dictionary[e]
    names[e] = elemento['direc_name']

lmaxes_configs = [(h, s, b)]

#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']
del estimators_dictionary

noisetag = data['noisekey']
trispectrumtag = data['trispectrumkey']
primarytag = data['primarykey']
secondarytag = data['secondarykey']
primarycrosstag = data['primarycrosskey']

kkkey = data['kkkey']
kgkey = data['kgkey']
ggkey = data['ggkey']
ellskey = data['ellskey']
thetakey = data['thetakey']
thetacrosskey = data['thetacrosskey']

totalabsbiaskey = data['totalabsbiaskey']
totalbiaskey = data['totalbiaskey']
sumalltotalabsbiaskey = data['sumalltotalabsbiaskey']
sumalltotalbiaskey = data['sumalltotalbiaskey']
sumallcrosstotalabsbiaskey = data['sumallcrosstotalabsbiaskey']
sumallcrosstotalbiaskey = data['sumallcrosstotalbiaskey']


lmin_sel, lmax_sel = data['lmin_sel'], data['lmax_sel']

optversion = data['optversion']

if noisebiasconstr:
    n_equals_b_dir = 'noiseequalsbias'
else:
    n_equals_b_dir = ''

if invvariance:
    inv_variance_dir = 'inversevariance'
else:
    inv_variance_dir = ''


bias_source = data['optimisation']['bias_source']

for fgnamefile in [fgnamefiles[0]]:
    for lmaxes in lmaxes_configs:
        lmaxes_dict = {}
        lmax_directory = ''
        for e_index, e in enumerate(estimators):
            l = lmaxes[e_index]
            lmaxes_dict[e] = l
            lmax_directory += f'{names[e]}{l}'

        print('Doing for', lmax_directory)
 
        P = PP/lmax_directory

        getoutname = lambda key: f'{key}_{nu}.npy'
        noises = np.load(P/getoutname(noisetag))

        getoutname2 = lambda key: f'{key}_total_{nu}.npy'

        if bias_source == 'total':
            biases = np.load(P/'total'/getoutname2(totalbiaskey)) #getoutname('sum_all_totalbias'))
            biasescross = np.load(P/'total'/getoutname2(primarycrosstag)) #/getoutname('sum_all_crosstotalbias'))
        elif bias_source == 'sum_bias':
            biases = np.load(P/getoutname(totalbiaskey))
            biasescross = np.load(P/getoutname(sum_all_crosstotalbias))
        elif bias_source == 'sum_abs_bias':
            biases = np.load(P/getoutname(sumalltotalabsbiaskey))
            biasescross = np.load(P/getoutname(sumallcrosstotalabsbiaskey))
          

        kg = np.load(P/getoutname(kgkey))
        kk = np.load(P/getoutname(kkkey))
        gg = np.load(P/getoutname(ggkey))
        ells = np.load(P/getoutname(ellskey))
        theta = np.load(P/getoutname(thetakey))
        thetacross = np.load(P/getoutname(thetacrosskey))
        estimators_to_ignore = 'hu_ok'
        Optimizerkk = best.Opt(estimators, lmin_sel, lmax_sel, ells, kk, theta, biases, noises, estimators_to_ignore)        
        result = Optimizerkk.optimize(optversion, method = 'diff-ev', gtol = gtol, bounds = [0., 1.], noisebiasconstr = noisebiasconstr, fb = fb, inv_variance = invvariance)

        result.save_all(pathlib.Path('twocases')/lmax_directory, f'auto_fb_{fb}')
        result.save(Optimizerkk.biases_selected, pathlib.Path('twocases')/lmax_directory, f'biases')        

