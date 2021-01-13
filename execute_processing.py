import pathlib

import re

import os

import argparse

import yaml

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

mock_numb = len(all_lmaxes_directories)

print('Number of directories for lmaxes', mock_numb)

iMax = mock_numb
iMin = 0

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
                os.system(f'python process_results.py {values_file} {fb} {neb} {inv_} {h} {s} {b}')            
