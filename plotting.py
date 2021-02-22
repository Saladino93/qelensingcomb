import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

csfont = {'fontname': 'serif'}
font = font_manager.FontProperties(family = 'serif',
                                   style = 'normal', size = 12)

import itertools

import numpy as np 

import pathlib


plt.rc('font', family = 'serif')

def plot_per_l(data, lmax_fixed, deltal, fsky, paperplots):

        analysis_directory = data['analysisdirectory']
        PP = pathlib.Path(analysis_directory)

        fgnamefiles = data['fgnamefiles']

        estimators_dictionary = data['estimators']
        estimators = list(estimators_dictionary.keys())

        Nsims = data['Nsims']

        noisetag = data['noisekey']
        trispectrumtag = data['trispectrumkey']
        primarytag = data['primarykey']
        secondarytag = data['secondarykey']
        primarycrosstag = data['primarycrosskey']

        lmin_sel, lmax_sel = data['lmin_sel'], data['lmax_sel']

        lista_lmaxes = []
        names = {}

        for e in estimators:
            elemento = estimators_dictionary[e]
            names[e] = elemento['direc_name']
            lmax_min, lmax_max = elemento['lmax_min'], elemento['lmax_max']
            num = elemento['number']
            lista_lmaxes += [np.linspace(lmax_min, lmax_max, num, dtype = int)]

        nu = estimators_dictionary[estimators[0]]['nu']
        
        colors_ests = {}
        labels_ests = {}
        for e in estimators:
            colors_ests[e] = estimators_dictionary[e]['color']
            labels_ests[e] = estimators_dictionary[e]['label']


        tri = 'Trispectrum'
        pri = 'Primary'
        pricross = 'Primary Cross'
        sec = 'Secondary'
        titles = [tri, pri, sec, pricross]

        titles_tags = {}
        titles_tags[tri] = trispectrumtag
        titles_tags[pri] = primarytag
        titles_tags[pricross] = primarycrosstag
        titles_tags[sec] = secondarytag


        foreground = {}
        foreground['tsz'] = 'tSZ'
        foreground['cib'] = 'CIB'
        foreground['ksz'] = 'kSZ'
        foreground['radiops'] = 'Radio'
        foreground['total'] = 'Sum'

        ylims = {}

        ylim = {}
        ylim['tsz'] = {}
        ylim['tsz']['ymin'] = -0.01
        ylim['tsz']['ymax'] = 0.1

        ylim['ksz'] = {}
        ylim['ksz']['ymin'] = -0.025
        ylim['ksz']['ymax'] = 0.025

        ylim['radiops'] = {}
        ylim['radiops']['ymin'] = -0.025
        ylim['radiops']['ymax'] = 0.025

        ylim['cib'] = {}
        ylim['cib']['ymin'] = -0.1
        ylim['cib']['ymax'] = 0.1

        ylim['total'] = {}
        ylim['total']['ymin'] = -0.1
        ylim['total']['ymax'] = 0.1

        ylims[tri] = ylim

        #######

        ylim = {}
        ylim['tsz'] = {}
        ylim['tsz']['ymin'] = -0.1
        ylim['tsz']['ymax'] = 0.1

        ylim['ksz'] = {}
        ylim['ksz']['ymin'] = -0.025
        ylim['ksz']['ymax'] = 0.025

        ylim['radiops'] = {}
        ylim['radiops']['ymin'] = -0.025
        ylim['radiops']['ymax'] = 0.1

        ylim['cib'] = {}
        ylim['cib']['ymin'] = -0.1
        ylim['cib']['ymax'] = 0.1

        ylim['total'] = {}
        ylim['total']['ymin'] = -0.12
        ylim['total']['ymax'] = 0.12

        ylims[pri] = ylim

        #######

        ylim = {}

        ylim['tsz'] = {}
        ylim['tsz']['ymin'] = -0.05
        ylim['tsz']['ymax'] = 0.1

        ylim['ksz'] = {}
        ylim['ksz']['ymin'] = -0.01
        ylim['ksz']['ymax'] = 0.01

        ylim['radiops'] = {}
        ylim['radiops']['ymin'] = -0.01
        ylim['radiops']['ymax'] = 0.01

        ylim['cib'] = {}
        ylim['cib']['ymin'] = -0.1
        ylim['cib']['ymax'] = 0.1

        ylim['total'] = {}
        ylim['total']['ymin'] = -0.12
        ylim['total']['ymax'] = 0.12

        ylims[pricross] = ylim

        #######

        ylim = {}
        ylim['tsz'] = {}
        ylim['tsz']['ymin'] = -0.05
        ylim['tsz']['ymax'] = 0.05

        ylim['ksz'] = {}
        ylim['ksz']['ymin'] = -0.05
        ylim['ksz']['ymax'] = 0.05

        ylim['radiops'] = {}
        ylim['radiops']['ymin'] = -0.05
        ylim['radiops']['ymax'] = 0.05

        ylim['cib'] = {}
        ylim['cib']['ymin'] = -0.1
        ylim['cib']['ymax'] = 0.1

        ylim['total'] = {}
        ylim['total']['ymin'] = -0.12
        ylim['total']['ymax'] = 0.12

        ylims[sec] = ylim

        Ne = len(estimators)
        lmaxes_list = [lmax_fixed for i in range(Ne)]
        lmaxes_list = [tuple(lmaxes_list)]
        for lmaxes in lmaxes_list:
            lmaxes_dict = {}
            lmax_directory = ''
            for e_index, e in enumerate(estimators):
                l = lmaxes[e_index]
                lmaxes_dict[e] = l
                lmax_directory += f'{names[e]}{l}'

            for t in titles:
                fig, ax = plt.subplots(nrows = len(fgnamefiles), ncols = 1, sharex = True, figsize = (8, 7))

                titolo = 'Relative bias on $C_L^{\kappa_{\mathrm{CMB}} \kappa_{\mathrm{CMB}}}$: '+f'{t}' if t != 'Primary Cross' else 'Relative bias on $C_L^{\kappa_{\mathrm{CMB}} g}$'
                fig.suptitle(titolo, fontsize = 20, **csfont)

                #primary = get(primarytag)
                #secondary = get(secondarytag)
                #trispectrum = get(trispectrumtag)

                #primaryscatter = getscatter(primarytag)
                #secondaryscatter = getscatter(secondarytag)
                #trispectrumscatter = getscatter(trispectrumtag)

                for fgindex, fgnamefile in enumerate(fgnamefiles):
                    
                    P = PP/lmax_directory
                    Pfg = PP/lmax_directory/fgnamefile

                    getoutname = lambda key: f'{key}_{fgnamefile}_{nu}.npy'
                    get = lambda key: np.load(Pfg/getoutname(key))

                    getoutnamescatter = lambda key: f'scatter_{key}_{fgnamefile}_{nu}.npy'
                    getscatter = lambda key: np.load(Pfg/getoutnamescatter(key))

                    noises = get(noisetag)
                    ells = get('ells')
                    kk = get('kk')
                    kg = get('kg')
                    gg = get('gg')
                
                    specificbias = get(titles_tags[t])
                    specificscatter = getscatter(titles_tags[t])

                    ylim = ylims[t]
                    
                    for e_index, e in enumerate(estimators):
                                        
                            
                        noise_k = noises[0, 0] #For QE
                        kktot = kk+noise_k
                        
                        if specificscatter.ndim > 2:
                            err = specificscatter[e_index, e_index]/kk
                            y = specificbias[e_index, e_index]/kk
                            
                            stat_uncert = np.sqrt(2/(2*ells*deltal*fsky))*kktot/kk
                        else:
                            err = specificscatter[e_index]/kg
                            y = specificbias[e_index]/kg
                            
                            stat_uncert = np.sqrt(1/(2*ells*deltal*fsky)*(kktot*gg+kg**2.))/kg
                        

                        ax[fgindex].plot(ells, y, color = colors_ests[e], label = labels_ests[e])
                        ax[fgindex].fill_between(ells, y-err, y+err, color = colors_ests[e], alpha = 0.4)
                        
                        ax[fgindex].fill_between(ells, -stat_uncert, stat_uncert, color = 'black', alpha = 0.05)
                        

                        ax[fgindex].set_ylim(ymin = ylim[fgnamefile]['ymin'], ymax = ylim[fgnamefile]['ymax'])
                        ax[fgindex].set_ylabel(f'{foreground[fgnamefile]}', size = 15)
                        ax[fgindex].set_ylim(ylim[fgnamefile]['ymin'], ylim[fgnamefile]['ymax'])
                        ax[fgindex].set_xlim(30, 2000)
                        ax[fgindex].axhline(y = 0, color = 'black', lw = 1)
                        ax[fgindex].tick_params(axis = 'y', which = 'major', labelsize = 15)
                        ax[fgindex].tick_params(axis = 'x', which = 'major', labelsize = 20)
                    

                ax[-1].set_xscale('log')
                ax[-1].set_xlabel('$L$', size = 20)
                
                ax[0].legend(loc = "best", ncol = len(estimators), prop = font)
                
                extra_title = nu
                #fig.tight_layout()
                fig.savefig(paperplots/f'biases_{extra_title}_{titles_tags[t]}_3500.png', bbox_inches = 'tight', dpi = 300)

