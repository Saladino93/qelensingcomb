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
all_lmaxes_directories =  [x.name for x in path.iterdir() if x.is_dir()]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = 0

mock_numb = len(all_lmaxes_directories)
delta = mock_numb/size

iMax = (rank+1)*delta+start
iMin = rank*delta+start

iMax = int(iMax)
iMin = int(iMin)

optdict = data['optimisation']

gtol = optdict['gtol']
fbs = optdict['fbs']
inv_variances = optdict['inv_variances']
noiseequalsbias = optdict['noiseequalsbias']



for inv_ in inv_variances:
    for neb in noiseequalsbias:
        for fb in fbs:
            for i in range(iMin, iMax):
                h, s, b = re.findall(r'\d+', all_lmaxes_directories[i])
                os.system(f'python lmax_optimize.py {values_file} {fb} {gtol} {neb} {inv_} {h} {s} {b}')            
