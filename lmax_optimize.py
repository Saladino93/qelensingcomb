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

my_parser.add_argument('lmaxes',
                       nargs = '+',
                       type = int)

args = my_parser.parse_args()

values_file = args.Configuration
fb = args.fb
gtol = args.gtol
noisebiasconstr = bool(args.noisebiasconstr)
invvariance = bool(args.invvariance)
lmaxes = args.lmaxes

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

lmaxes_configs = [tuple(lmaxes)]

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


def get_est_weights(Opt, index):
    '''
    index = 0, 1, ....
    e.g. h, s, b -> index = 1 gives s
    '''
    Nest = len(Opt.estimators)
    nbins = Opt.nbins
    zeros = np.zeros(Nest*nbins)
    for j in range(nbins):
        zeros[index+Nest*j:index+(Nest*j+1)] = 1.
    return zeros


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

        Optimizerkk = best.Opt(estimators, lmin_sel, lmax_sel, ells, kk, theta, biases, noises)        
        result = Optimizerkk.optimize(optversion, method = 'diff-ev', gtol = gtol, bounds = [0., 1.], noisebiasconstr = noisebiasconstr, fb = fb, inv_variance = invvariance)

        result.save_all(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'auto_fb_{fb}')
        result.save(Optimizerkk.biases_selected, pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'biases')        
        result.save(Optimizerkk.noises_selected, pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'noises')
        fnb_getter = lambda Opt, fb_val, invvar: Opt.get_f_n_b(Opt.ells_selected, Opt.theory_selected, Opt.theta_selected, Opt.biases_selected,
                              sum_biases_squared = False, bias_squared = False, fb = fb_val, inv_variance = invvar)
         
        Nestimators = len(Optimizerkk.estimators) 
        results_array = np.zeros((3, Nestimators+1))
        for index in range(Nestimators):
            x_estimator = get_est_weights(Optimizerkk, index = index)
            f, n, b = fnb_getter(Optimizerkk, fb, True)
            f_estimator, n_estimator, b_estimator = f(x_estimator), n(x_estimator), b(x_estimator)  
            results_array[:, index+1] = np.array([f_estimator, n_estimator, b_estimator])

        f, n, b = fnb_getter(Optimizerkk, fb, invvariance)
        fcomb, ncomb, bcomb = f(result.x), n(result.x), b(result.x)
        results_array[:, 0] = np.array([fcomb, ncomb, bcomb])

        results.save(results_array, pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'alens')
        
        Optimizerkg = best.Opt(estimators, lmin_sel, lmax_sel, ells, kg, thetacross, biasescross, noises)
        result = Optimizerkg.optimize(optversion, method = 'diff-ev', gtol = gtol, bounds = [0., 1.], noisebiasconstr = noisebiasconstr, fb = fb, inv_variance = invvariance)

        result.save_all(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'cross_fb_{fb}')
        result.save(Optimizerkg.biases_selected, pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'cross_biases')
