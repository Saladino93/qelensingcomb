import argparse

import pathlib

import sys

import utilities as u

import yaml

import numpy as np

import itertools

from mpi4py import MPI

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

estimatorssubset = data['estimatorssubset']

lista_lmaxes = []

names = {}

if estimatorssubset == '':
    estimatorssubset = estimators
else:
    print(f'Estimators subset {estimatorssubset}')

for e in estimators:
    elemento = estimators_dictionary[e]
    names[e] = elemento['direc_name']
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    if e not in estimatorssubset:
        num = 1
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]


lmaxes_configs_input = data['lmaxes_configs_input_to_try']
if lmaxes_configs_input:
    lmaxes_configs = [tuple([l for e in estimators]) for l in lista_lmaxes[0]]
else:
    lmaxes_configs = list(itertools.product(*lista_lmaxes))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mock_numb = len(lmaxes_configs)
delta = mock_numb/size
start = 0
iStop = mock_numb

rank_ass = rank

iMax = (rank_ass+1)*delta+start
iMin = rank_ass*delta+start

iMax = int(iMax)
iMin = int(iMin)

if iMax > iStop:
	iMax = iStop
elif (iMax > (iStop - delta)) and iMax <iStop:
	iMax = iStop

if size == 1:
    iMin = 0
    iMax = iStop

lmaxes_configs = lmaxes_configs[iMin:iMax] 



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

#given an lmax_directory, creates all the necessary dictionaries
noisedicttag = data['noisekey']
trispectrumdicttag = data['trispectrumkey']
primarydicttag = data['primarykey']
secondarydicttag = data['secondarykey']
primarycrossdicttag = data['primarycrosskey']

validationtag = 'val'

kkkey = data['kkkey']
kgkey = data['kgkey']
ggkey = data['ggkey']
ellskey = data['ellskey']

savingdirectory = data['savingdirectory']
spectra_path = data['spectra_path']
sims_directory = data['sims_directory']

def get_dict_element(i, nu, fgnamefile, estA, lmaxA, estB, lmaxB):
    temp_directory = names[estA]+str(lmaxA)+names[estB]+str(lmaxB)
    dic = u.dictionary(pathlib.Path(savingdirectory)/'all'/fgnamefile, temp_directory) #NOTE: add fgnamefile
    dic = dic.read(f'{fgnamefile}_{nu}_{i}')
    return dic

def save_dict(fgnamefile, nu, estimators, lconfig):
    noisetemp = {} 
    for i in range(Nsims):
            dictionary = u.dictionary(pathlib.Path(savingdirectory), lmax_directory)
            dictionary.create_subdictionary(noisedicttag)
            dictionary.create_subdictionary(trispectrumdicttag)
            dictionary.create_subdictionary(primarydicttag)
            dictionary.create_subdictionary(secondarydicttag)
            dictionary.create_subdictionary(primarycrossdicttag)
            dictionary.create_subdictionary(validationtag)
             
            estimatorcombs = list(itertools.combinations_with_replacement(list(estimators), 2))
            if lmaxes_configs_input:
                estimatorcombs = [(e, e) for e in estimators]
            for estA, estB in estimatorcombs:
                lmaxA, lmaxB = lconfig[estimators.index(estA)], lconfig[estimators.index(estB)]
                dic = get_dict_element(i, nu, fgnamefile, estA, lmaxA, estB, lmaxB) 
                tag_gal = f'{primarycrossdicttag}-{estA}'
                if not dictionary.exists_in_subdictionary(primarycrossdicttag, tag_gal): 
                    dictionary.add_to_subdictionary(primarycrossdicttag, tag_gal, dic[primarycrossdicttag][tag_gal])
                
                tag = f'{trispectrumdicttag}-{estA}-{estB}'
                dictionary.add_to_subdictionary(trispectrumdicttag, tag, dic[trispectrumdicttag][tag])
                tag = f'{primarydicttag}-{estA}-{estB}'
                dictionary.add_to_subdictionary(primarydicttag, tag, dic[primarydicttag][tag])
                tag = f'{secondarydicttag}-{estA}-{estB}'
                dictionary.add_to_subdictionary(secondarydicttag, tag, dic[secondarydicttag][tag])

                tag = f'{noisedicttag}-{estA}-{estB}'
                if i == 0:
                    elemento = dic[noisedicttag][tag]
                    noisetemp[tag] = elemento
                else:
                    elemento = noisetemp[tag]
                    
                dictionary.add_to_subdictionary(noisedicttag, tag, elemento)

                valtag = f'{validationtag}-{estA}'
                if not dictionary.exists_in_subdictionary(validationtag, valtag):
                    dictionary.add_to_subdictionary(validationtag, valtag, dic[validationtag][valtag])
            
            dictionary.add(ggkey, dic[ggkey])
            dictionary.add(kkkey, dic[kkkey])
            dictionary.add(kgkey, dic[kgkey])
            dictionary.add(ellskey, dic[ellskey])

            dictionary.save(f'{fgnamefile}_{nu}_{i}')



for lconfig in lmaxes_configs:
    ndirs += 1
    lmax_directory = ''

    lmax_directory_out = ''



    for e_index, e in enumerate(estimators):
        l = lconfig[e_index]
        lmax_directory += f'{names[e]}{l}'

    for e_index, e in enumerate(estimatorssubset):
        l = lconfig[e_index]
        lmax_directory_out += f'{names[e]}{l}'

    print('In directory', lmax_directory, 'Out directory', lmax_directory_out)

    PPP = PP/lmax_directory_out
    if not PPP.exists():
        PPP.mkdir(parents = True, exist_ok = True)

    for fgnamefile in fgnamefiles:
        
        #first, process
        save_dict(fgnamefile, nu, estimators, lconfig)
        
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
                array = u.get_element(dictionary[k], estimatorssubset)
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
        totalbiasscatter = 0.
        for tag in biasestags:
            totalbias += np.load(P/getoutname(tag))
        total = []
        for j in range(Nsims):
            array = 0.
            for tag in biasestags:
                dictionary = dic.read(f'{fgnamefile}_{nu}_{j}')
                array += u.get_element(dictionary[tag], estimators)
            total += [array]
        
        mean, totalbiasscatter = u.get_mean_and_scatter(Nsims, total)
        factor = np.sqrt(A/A_octanct) if oldversion else 1.
        totalbiasscatter *= factor
        #print(mean/totalbias)
        np.save(P/getoutname(totalbiaskey), totalbias)
        np.save(P/f'scatter_{totalbiaskey}_{nu}.npy', totalbiasscatter)

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

    
