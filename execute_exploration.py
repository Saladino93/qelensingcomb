import argparse

import yaml

from mpi4py import MPI

import pathlib

import re

import os

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

output = data['analysisdirectory']

path = pathlib.Path(output)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = 0

all_configs = []
all_its = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
all_scales =  [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

all_crosses = [0.75, 0.85, 0.9, 0.92, 0.95, 1.0]

for i in all_its:
    for s in all_scales:
        for c in all_crosses:
            all_configs += [(i, s, c)]

mock_numb = len(all_configs)
delta = mock_numb/size

iMax = (rank+1)*delta+start
iMin = rank*delta+start

iMax = int(iMax)
iMin = int(iMin)

if iMax > mock_numb:
	iMax = mock_numb
elif (iMax >= (mock_numb - delta)) and iMax < mock_numb:
	iMax = mock_numb

configs = all_configs[iMin:iMax]

optdict = data['optimisation']

gtol = optdict['gtol']
fbs = optdict['fbs']
inv_variances = optdict['inv_variances']
noiseequalsbias = optdict['noiseequalsbias']


for item in configs:
    iteration, scale, cross = item
    os.system(f'python explore_diff_ev.py configurations/configILCnewreg.yaml 1.0 3500 0 0 {scale} {iteration} {cross} 4000 3500 4500 4000')
