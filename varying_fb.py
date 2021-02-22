import os

values_file = 'configurations/config.yaml'

gtol = 2000
fbs = [0., 0.01, 0.05, 0.1, 0.5, 1., 2., 5., 10., 100]

neb = 0
inv_ = 0

h, s, b = 3500, 3500, 3500

for fb in fbs:
    os.system(f'python lmax_optimize.py {values_file} {fb} {gtol} {neb} {inv_} {h} {s} {b}')
