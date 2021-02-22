import argparse

import sys

import pathlib

import yaml

import numpy as np

import itertools

import utilities as u

import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

csfont = {'fontname':'cmss10'}
font = font_manager.FontProperties(family='cmss10',
                                   style='normal', size = 12)

import best
import re


import plotting


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

savingdirectory = data['savingdirectory']

results_directory = data['resultsdirectory']
spectra_path = data['spectra_path']
sims_directory = data['sims_directory']
WR = u.write_read(sims_directory)

PP = pathlib.Path(analysis_directory)
Pplots = pathlib.Path(plots_directory)

fgnamefiles = data['fgnamefiles']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())

Nsims = data['Nsims']

lista_lmaxes = []

names = {}

for e in estimators:
    elemento = estimators_dictionary[e]
    names[e] = elemento['direc_name']
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]

lmaxes_configs = list(itertools.product(*lista_lmaxes))


#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']

noisetag = data['noisekey']
trispectrumtag = data['trispectrumkey']
primarytag = data['primarykey']
secondarytag = data['secondarykey']
primarycrosstag = data['primarycrosskey']

lmin_sel, lmax_sel = data['lmin_sel'], data['lmax_sel']

optversion = data['optversion']

fsky = 1.

lEdges = np.logspace(np.log10(10), np.log10(4000), 15, 10.) #NOTE
deltal = lEdges[1:]-lEdges[:-1]

paperplots = pathlib.Path('paperplots')

lmax_fixed = 3500
extra_title = ''
plotting.plot_per_l(data, lmax_fixed, deltal, fsky, paperplots)




