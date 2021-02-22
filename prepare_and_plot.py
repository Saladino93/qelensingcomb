import argparse

import pathlib

import sys

import utilities as u

import yaml

import numpy as np

import itertools

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

PP = pathlib.Path(analysis_directory)
if not PP.exists():
    PP.mkdir(parents = True, exist_ok = True)

Pplots = pathlib.Path(plots_directory)
if not Pplots.exists():
    Pplots.mkdir(parents = True, exist_ok = True)

fgnamefiles = data['fgnamefiles']

oldversion = data['oldversion']

Nsims = data['Nsims']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())
print(f'Estimators {estimators}')
#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']

lista_lmaxes = []

names = {}

for e in estimators:
    elemento = estimators_dictionary[e]
    names[e] = elemento['direc_name']
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]


lmaxes_configs_input = data['lmaxes_configs_input_to_try']
if len(lmaxes_configs_input) > 0:
    lmaxes_configs = lmaxes_configs_input
else:
    lmaxes_configs = list(itertools.product(*lista_lmaxes))

del estimators_dictionary


noisetag = data['noisekey']
trispectrumtag = data['trispectrumkey']
primarytag = data['primarykey']
secondarytag = data['secondarykey']
primarycrosstag = data['primarycrosskey']

biasestags = [trispectrumtag, primarytag, secondarytag]

totalabsbiaskey = data['totalabsbiaskey']
totalbiaskey = data['totalbiaskey']
sumalltotalabsbiaskey = data['sumalltotalabsbiaskey']
sumalltotalbiaskey = data['sumalltotalbiaskey']
sumallcrosstotalabsbiaskey = data['sumallcrosstotalabsbiaskey']
sumallcrosstotalbiaskey = data['sumallcrosstotalbiaskey']

kkkey = data['kkkey']
kgkey = data['kgkey']
ggkey = data['ggkey']
ellskey = data['ellskey']
thetakey = data['thetakey']
thetacrosskey = data['thetacrosskey']


#NOTE
function = lambda x: abs(x)

ndirs = 0

for lconfig in lmaxes_configs:
    ndirs += 1
    lmax_directory = ''
    for e_index, e in enumerate(estimators):
        l = lconfig[e_index]
        lmax_directory += f'{names[e]}{l}'
    print(lmax_directory)

    PPP = PP/lmax_directory
    if not PPP.exists():
        PPP.mkdir(parents = True, exist_ok = True)

    for fgnamefile in fgnamefiles:
        
        P = PPP/fgnamefile
        
        if not P.exists():
            P.mkdir(parents = True, exist_ok = True)

        dic = u.dictionary(directory, lmax_directory)
   
        i = 0 
        dictionary_temp = dic.read(f'{fgnamefile}_{nu}_{i}')
    
        #Loop over the elements of the saved dictionary
        for k in dictionary_temp.keys():
        
            total = []
        
            for j in range(Nsims):
                dictionary = dic.read(f'{fgnamefile}_{nu}_{j}')
                array = u.get_element(dictionary[k], estimators)
                total += [array]
        
            total = np.array(total)
        
            mean, scatter = u.get_mean_and_scatter(Nsims, total)
                        
            getoutname = lambda key: f'{key}_{fgnamefile}_{nu}.npy'
            np.save(P/getoutname(k), mean)
        
            els = [ellskey]
            if not k in els:
                outname = f'scatter_{k}_{fgnamefile}_{nu}.npy'
                #NOTE, correction for the old Manus and Simos sims
                A, A_octanct = 81*20**2, 5156.6
                factor = np.sqrt(A/A_octanct) if oldversion else 1.
                np.save(P/outname, scatter*factor)

        noises = np.load(P/getoutname(noisetag))
        kg = np.load(P/getoutname(kgkey))
        kk = np.load(P/getoutname(kkkey))
        gg = np.load(P/getoutname(ggkey))
        ells = np.load(P/getoutname(ellskey))

        theta = u.getcovarianceauto(noises, kk, fsky = 1.0)
        thetacross = u.getcovariancecross(noises, kk, kg, gg)
        
        np.save(P/getoutname(thetakey), theta)
        np.save(P/getoutname(thetacrosskey), thetacross)

        totalbias = 0.
        for tag in biasestags:
            totalbias += function(np.load(P/getoutname(tag)))
        np.save(P/getoutname(totalabsbiaskey), totalbias)

        totalbias = 0.
        for tag in biasestags:
            totalbias += np.load(P/getoutname(tag))
        np.save(P/getoutname(totalbiaskey), totalbias)

    totalabsbias = 0. 
    totalbias = 0.
    totalbiascross = 0.
    totalabsbiascross = 0.
 
    fgnamefilescopy = fgnamefiles.copy() 
    if 'total' in fgnamefilescopy: 
        fgnamefilescopy.remove('total') 
    for fgnamefile in fgnamefilescopy:
        P = PPP/fgnamefile
        totalabsbias += np.load(P/getoutname(totalabsbiaskey)) 
        totalbias += np.load(P/getoutname(totalbiaskey)) 
        totalbiascross += np.load(P/getoutname(primarycrosstag))
        totalabsbiascross += abs(np.load(P/getoutname(primarycrosstag)))
        
    getoutname = lambda key: f'{key}_{nu}.npy'
 
    nome = sumalltotalabsbiaskey+f'_{nu}.npy'
    #SUM OF FOREGROUND BIASES, WHERE BIAS IS THE SUM OF ABS(T)+ABS(P)+ABS(S) 
    np.save(PPP/nome, totalabsbias) 

    nome = sumalltotalbiaskey+f'_{nu}.npy'
    #SUM OF FOREGROUND BIASES, WHERE BIAS IS THE SUM OF T+P+S 
    np.save(PPP/nome, totalbias) 
    
    nome = sumallcrosstotalabsbiaskey+f'_{nu}.npy'    
    np.save(PPP/nome, totalabsbiascross)
    nome = sumallcrosstotalbiaskey+f'_{nu}.npy'
    np.save(PPP/nome, totalbiascross)    


    np.save(PPP/getoutname(thetakey), theta)
    np.save(PPP/getoutname(thetacrosskey), thetacross)
    np.save(PPP/getoutname(kgkey), kg) 
    np.save(PPP/getoutname(kkkey), kk)
    np.save(PPP/getoutname(ggkey), gg)
    np.save(PPP/getoutname(noisetag), noises)    
    np.save(PPP/getoutname(ellskey), ells)

print(f'Total number of direcs ', ndirs)

    
