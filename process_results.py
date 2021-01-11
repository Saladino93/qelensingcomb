import argparse

import pathlib

import sys

import yaml

import numpy as np

import best

import itertools

import utilities as u

my_parser = argparse.ArgumentParser(description = 'Configuration file.')

my_parser.add_argument('Configuration',
                       metavar='configuration file',
                       type = str,
                       help = 'the path to configuration file')

my_parser.add_argument('fb',
                       metavar='bias enhancement',
                       type = float)

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
noisebiasconstr = bool(args.noisebiasconstr)
invvariance = bool(args.invvariance)
h = args.h
s = args.s
b = args.b

#print('Applying invvariance', invvariance)

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


fnb_getter = lambda Opt, fb_val, invvar: Opt.get_f_n_b(Opt.ells_selected, Opt.theory_selected, Opt.theta_selected, Opt.biases_selected, 
                              sum_biases_squared = False, bias_squared = False, fb = fb_val, inv_variance = invvar)


def get_est_weights(Opt, index, invvar):
    '''
    index = 0, 1, ....
    e.g. h, s, b -> index = 1 gives s
    '''
    nbins = Opt.nbins
    zeros = np.zeros(3*nbins)
    for j in range(nbins):
        zeros[index+3*j:index+(3*j+1)] = 1.
    return zeros


def get_dict_results(fgnamefile, lmax_directory, fb):

    P = PP/lmax_directory

    getoutname = lambda key: f'{key}_{nu}.npy'
    noises = np.load(P/getoutname(noisetag))

    biases = np.load(P/getoutname('sum_all_totalabsbias'))
    biasescross = np.load(P/getoutname('sum_all_crosstotalabsbias'))

    kg = np.load(P/getoutname('kg'))
    kk = np.load(P/getoutname('kk'))
    gg = np.load(P/getoutname('gg'))
    ells = np.load(P/getoutname('ells'))
    theta = np.load(P/getoutname('theta'))
    thetacross = np.load(P/getoutname('thetacross'))

    Optimizerkk = best.Opt(estimators, lmin_sel, lmax_sel, ells, kk, theta, biases, noises)
    resultkk = best.Res()
    resultkk.load_all(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'auto_fb_{fb}')
    biases = resultkk.load(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'biases')

    directory_of_saving = pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir
    dic = u.dictionary(directory_of_saving)
 

    f, n, b = fnb_getter(Optimizerkk, fb, invvariance)
    finv, ninv, binv = fnb_getter(Optimizerkk, fb_val = 0, invvar = True)

    xhuok = get_est_weights(Optimizerkk, index = 0, invvar = invvariance)
    xshear = get_est_weights(Optimizerkk, index = 1, invvar = invvariance)
    xbh = get_est_weights(Optimizerkk, index = 2, invvar = invvariance) 

    autotag = 'auto'
    crosstag = 'cross'

    dic.create_subdictionary(autotag)
    dic.create_subdictionary(crosstag)

    dic.add_to_subdictionary(autotag, 'noise', n(resultkk.x))
    dic.add_to_subdictionary(autotag, 'bias', b(resultkk.x))
    dic.add_to_subdictionary(autotag, 'total', f(resultkk.x))

    dic.add_to_subdictionary(autotag, 'noisehuok', ninv(xhuok))
    dic.add_to_subdictionary(autotag, 'biashuok', binv(xhuok))

    dic.add_to_subdictionary(autotag, 'noiseshear', ninv(xshear))
    dic.add_to_subdictionary(autotag, 'biasshear', binv(xshear))

    dic.add_to_subdictionary(autotag, 'noisebh', ninv(xbh))
    dic.add_to_subdictionary(autotag, 'biasbh', binv(xbh))
    
    dic.add_to_subdictionary(autotag, 'wh', resultkk.ws[:, 0])
    #print(resultkk.ws[:, 0])   
    dic.add_to_subdictionary(autotag, 'ws', resultkk.ws[:, 1])
    dic.add_to_subdictionary(autotag, 'wbh', resultkk.ws[:, 2])
    dic.add_to_subdictionary(autotag, 'wl', resultkk.ws[:, -1])

    dic.add_to_subdictionary(autotag, 'biases', Optimizerkk.biases_selected)
    dic.add_to_subdictionary(autotag, 'noises', Optimizerkk.noises_selected)

    dic.add('ells', resultkk.ells)
       

    Optimizerkg = best.Opt(estimators, lmin_sel, lmax_sel, ells, kg, thetacross, biasescross, noises)
    resultkg = best.Res()
    resultkg.load_all(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'cross_fb_{fb}')
    biases_cross = resultkk.load(pathlib.Path(results_directory)/lmax_directory/inv_variance_dir/n_equals_b_dir, f'cross_biases')

    f, n, b = fnb_getter(Optimizerkg, fb, invvariance)
    finv, ninv, binv = fnb_getter(Optimizerkg, fb_val = 0, invvar = True)   

    
    dic.add_to_subdictionary(crosstag, 'noise', n(resultkg.x))
    dic.add_to_subdictionary(crosstag, 'bias', b(resultkg.x))
    dic.add_to_subdictionary(crosstag, 'total', f(resultkg.x))

    dic.add_to_subdictionary(crosstag, 'noisehuok', ninv(xhuok))
    dic.add_to_subdictionary(crosstag, 'biashuok', binv(xhuok))

    dic.add_to_subdictionary(crosstag, 'noiseshear', ninv(xshear))
    dic.add_to_subdictionary(crosstag, 'biasshear', binv(xshear))

    dic.add_to_subdictionary(crosstag, 'noisebh', ninv(xbh))
    dic.add_to_subdictionary(crosstag, 'biasbh', binv(xbh))
    
    dic.add_to_subdictionary(crosstag, 'wh', resultkg.ws[:, 0])
    dic.add_to_subdictionary(crosstag, 'ws', resultkg.ws[:, 1])
    dic.add_to_subdictionary(crosstag, 'wbh', resultkg.ws[:, 2])
    dic.add_to_subdictionary(crosstag, 'wl', resultkg.ws[:, -1])

    dic.add_to_subdictionary(crosstag, 'biases', Optimizerkg.biases_selected)
    dic.add_to_subdictionary(crosstag, 'noises', Optimizerkg.noises_selected)

    dic.save(f'results_fb_{fb}')


for fgnamefile in [fgnamefiles[0]]:
    for lmaxes in lmaxes_configs:
        lmaxes_dict = {}
        lmax_directory = ''
        for e_index, e in enumerate(estimators):
            l = lmaxes[e_index]
            lmaxes_dict[e] = l
            lmax_directory += f'{names[e]}{l}'

        print('Doing for', lmax_directory)
 
        get_dict_results(fgnamefile, lmax_directory, fb) 
