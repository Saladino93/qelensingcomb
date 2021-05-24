import symlens as s
from pixell import enmap, utils as putils, powspec
from scipy.interpolate import interp1d
import numpy as np
from orphics import lensing,io,cosmology,maps
import yaml
import pathlib
import utilities as u
import itertools

values_file = 'configurations/configILC_plotting_validation_check.yaml' 
with open(values_file, 'r') as stream:
            data = yaml.safe_load(stream)



plots_directory = data['plotsdirectory']

analysis_directory = data['analysisdirectory']

savingdirectory = data['savingdirectory']

Nsims = data['Nsims']

results_directory = data['resultsdirectory']
spectra_path = data['spectra_path']
sims_directory = data['sims_directory']
WR = u.write_read(sims_directory)

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
    lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
    num = elemento['number']
    lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]

lmaxes_configs = list(itertools.product(*lista_lmaxes))


#CHOOSE nu
nu = estimators_dictionary[estimators[0]]['nu']

estimators_dictionary = data['estimators']
estimators = list(estimators_dictionary.keys())
estimatorcombs = list(itertools.combinations_with_replacement(list(estimators), 2))

logmode = data['logmode']
nlogBins = data['nlogBins']
deltal = data['deltalplot']

lmaxes_dict = {}
lmax_directory = ''
for e_index, e in enumerate(estimators):
    l = 3500
    lmaxes_dict[e] = l
    lmax_directory += f'{names[e]}{l}'




pols = ['TE', 'EE', 'EB'] #['TT','EE','TE','EB'] #,'TB']

field_names_A = ['ilcA', 'ilcA']

estA = 'hu_ok'

nuA = estimators_dictionary[estA]['nu']
lmax_A = lmaxes_dict[estA]

mapsObjA = u.mapNamesObj(nuA)

hardening_A = estimators_dictionary[estA]['hardening']

field_names_A = estimators_dictionary[estA]['field_names']

tszprofileA = estimators_dictionary[estA]['tszprofile']
tszprofile_A = None if tszprofileA == '' else 1.

changemap = lambda x: enmap.enmap(x, wcs)
C = u.Converting()
LoadA = u.LoadfftedMaps(mapsObj = mapsObjA, WR = WR, ConvertingObj = C, changemap = changemap, getfft = u.fft, lmax = lmax_A)

shape = LoadA.read_shape()
lonCenter, latCenter = 0, 0
shape, wcs = enmap.geometry(shape = shape, res = 1.*putils.arcmin, pos = (lonCenter, latCenter))
modlmap = enmap.modlmap(shape, wcs)

estimator_to_harden_A = 'hu_ok' if (estA in ['bh', 'pbh']) else estA

estimator_to_harden_A = 'symm' if ('symm' in estA) else estimator_to_harden_A #in ['symmbh', 'symmpbh']) else estA

#Get shape and wcs
shape = LoadA.read_shape()
lonCenter, latCenter = 0, 0
shape, wcs = enmap.geometry(shape = shape, res = 1.*putils.arcmin, pos = (lonCenter, latCenter))
modlmap = enmap.modlmap(shape, wcs)

feed_dict = u.Loadfeed_dict(pathlib.Path(spectra_path), field_names_A, field_names_A, modlmap, hardening_A, hardening_A, tszprofile_A, tszprofile_A)

nncalpolonly = {}

Lmin, Lmax = 20, 6000
lmin_A, lmax_A = 30, 4000

Binner = u.Binner(shape, wcs, lmin = 10, lmax = 6000, deltal = 40, log = False, nBins = 20)


for XYA in pols:
    A = u.Estimator(shape, wcs, feed_dict, estA, lmin_A, lmax_A,
                    field_names = field_names_A, groups = None, Lmin = Lmin, Lmax = Lmax,
                    hardening = hardening_A, estimator_to_harden = estimator_to_harden_A, XY = XYA) 
    NAB_cross = A.get_Nl_cross_other(feed_dict, A, tipo = 't') #A.get_Nl_cross(B)
    l, NAB_cross_binned = Binner.bin_spectra(NAB_cross)
    nncalpolonly[XYA+XYA] = NAB_cross_binned

    np.savetxt(f'polweights/{XYA}_{lmax_A}.txt', np.c_[l, NAB_cross_binned])
