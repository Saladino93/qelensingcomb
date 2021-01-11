import pathlib

import re

import os

path = pathlib.Path('output/')
all_lmaxes_directories =  [x.name for x in path.iterdir() if x.is_dir()]

mock_numb = len(all_lmaxes_directories)

print('Number of directories for lmaxes', mock_numb)

iMax = mock_numb
iMin = 0

iMax = int(iMax)
iMin = int(iMin)

fbs = [0, 1]#[0, 0.01, 0.03, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100]
inv_variances = [0, 1]
noiseequalsbias = [0]

for inv_ in inv_variances:
    for neb in noiseequalsbias:
        for fb in fbs:
            for i in range(iMin, iMax):
                h, s, b = re.findall(r'\d+', all_lmaxes_directories[i])
                os.system(f'python process_results.py config.yaml {fb} {neb} {inv_} {h} {s} {b}')            
