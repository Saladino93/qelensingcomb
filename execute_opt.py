from mpi4py import MPI

import pathlib

import re

import os

path = pathlib.Path('output/')
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

gtol = 10000
fbs = [0.01, 0.03, 0.05, 0.1, 0.5, 2, 5, 10, 100, 0.]
inv_variances = [1]
noiseequalsbias = [0]

#TODO
#Put list of fbs, invvar, gtol, noiseeqb, in config
#take config as input with argparser

for inv_ in inv_variances:
    for neb in noiseequalsbias:
        for fb in fbs:
            for i in range(iMin, iMax):
                h, s, b = re.findall(r'\d+', all_lmaxes_directories[i])
                os.system(f'python lmax_optimize.py configsumfgs.yaml {fb} {gtol} {neb} {inv_} {h} {s} {b}')            
