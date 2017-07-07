#!/usr/bin/env python

import numpy as np
from numpy import squeeze as sq
from pylab import *
import matplotlib.pyplot as plt
from math import *
import pickle
import mpl_toolkits.basemap as bm
#import scipy.io
from met_utils import *
#import plot_tools as pt
from netCDF4 import Dataset
import netCDF4 as netcdf4
from scipy.cluster.vq import *
from scipy.interpolate import griddata
from calendar import monthrange
from scipy import stats
import time
import copy 
import random
#
sys.path.append('/home/mcooke/Research/KMeans/')
from kmeans_functions import *
from merra_latlon import merra_latlon_toolik
from merra_process import *
from merra_opendap_tools import *
startstart = time.time()

## Initialize: User input ##
make = 1
makeeof = 1
saveclim=1
saveproc=1

clusteranom = 1
anomtype = 'd' # Use s for seasonal, d for daily
makecent = 1
plotdurn = 1
clusterplot = 1
distplot = 1
plot_boxplot = 0
makepc = 0
makeclass = 0
pcatest = 0
varname = 'H500'
k = 7
yeari = 1979
yearf = 2015
yearlist = range(yeari,yearf+1)
yeari2 = 1998
yearf2 = 2015
yearlist2 = range(yeari2,yearf2+1)
monthi = 6
monthf = 8
monthlist = range(monthi,monthf+1) 
lowres = 1
timefilter = 0 # must be even 
dayskip = 0
eofnum = 18 #33 #18 #7 
savenameexts = str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf) # Short save name extension
#savenameext = str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'_lowres_10sm'
savenameext = varname + '_' + savenameexts
#if lowres == 1:
    #savenameext = savenameext + '_lowres'
if timefilter > 0:
    timefilt_name = str(timefilter) + 'sm'
    savenameext = savenameext + '_' + timefilt_name
    tfolder = timefilt_name
else:
    timefilt_name = 'nosm'
    savenameext = savenameext + '_' + timefilt_name
    tfolder = timefilt_name
if dayskip > 0:
    dayskip_name = str(dayskip) + 'day'
    savenameext = savenameext + '_' + dayskip_name
    dfolder = dayskip_name
else:
    dayskip_name = 'daily'
    savenameext = savenameext + '_' + dayskip_name
    dfolder = dayskip_name
if eofnum > 0:
    eof_name = intto2dstr(eofnum) + 'eof'
    savenameext = savenameext + '_' + eof_name
    efolder = eof_name
else:
    eof_name = 'noeof'  
    savenameext = savenameext + '_' + eof_name
    efolder = eof_name

#savenameext_eof = savenameext
savenameext_nok = savenameext + '_' + anomtype

if clusteranom == 1:
    savenameext = savenameext + '_' + anomtype + 'anom'
savenameext = savenameext + '_k' + str(k)

imgfolder = tfolder + '_' + dfolder + '_' + efolder +'/'
if clusteranom == 1:
    imgfolder = imgfolder + 'anom/'
folderext = tfolder + '_' + dfolder + '_' + efolder

datadir_in = '/home/mcooke/Research/KMeans/'
datadir_out = '/home/mcooke/Research/KMeans/NewAlgorithm/NewMerra/ProcessedData/'

print 'Processing data for ', savenameext

###############################################

## Load lat/lon ##

loni, lati, nnx, nny, nn = merra_latlon_toolik()

##  Load data ##

allmonthlen = 0
for month in monthlist:
    allmonthlen+=monthrange(1999,month)[1]
daylen = 1
nt = len(yearlist)*allmonthlen*daylen
nty = allmonthlen*daylen
H500A = np.zeros((nt,nny,nnx))
data = np.zeros((nt,nn))
alltimes = np.zeros(nt)
iter1=0
iter2=nty
start = time.time()
if varname == 'H500':
    #fnamesave = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
    #fnamesave2 = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'
    fnamesave = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-2013.pickle'
    fnamesave_time = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-2013.pickle'
    fnamesave2 = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_2014-'+str(yearf)+'.pickle'
    fnamesave2_time = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_2014-'+str(yearf)+'.pickle'
else:
    fnamesave = datadir_in + 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
    fnamesave2 = datadir_in + 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'

with open(fnamesave,'rb') as fsave:
    H500A = pickle.load(fsave)
with open(fnamesave2,'rb') as fsave:
    H500A2 = pickle.load(fsave)
with open(fnamesave_time,'rb') as fsave:
    alltimes = pickle.load(fsave)
with open(fnamesave2_time,'rb') as fsave:
    alltimes2 = pickle.load(fsave)
H500A = np.concatenate((H500A, H500A2))
alltimes = np.concatenate((alltimes, alltimes2))

t = np.shape(H500A)[0]
for ti in range(t):
    data[ti,:] = np.reshape(H500A[ti,:,:], nn)


## Process data ## 

# Adjust data to be low res
H500Alr = H500A[:,::4,::4]
lonlr = loni[::4]
latlr = lati[::4]
# Reshape
nly, nlx = np.shape(H500Alr)[1::]
nl = nlx*nly
datalr = np.zeros((nt,nl))
for i in range(nt):
    datalr[i,:] = np.reshape(H500Alr[i,:,:], nl)
#datalro = np.copy(datalr)
lonip, latip = meshgrid(lonlr,latlr)
lonipf = np.reshape(lonip,nl)
latipf = np.reshape(latip,nl)
lonip, latip = meshgrid(lonlr,latlr)

# Detrend / make anomaly
data, clim, trend = detrend_anomaly(datalr, yearlist, monthlist, anomtype, clusteranom) 

if saveclim:
    # Save climatology data
    f_datac = open(datadir_out+anomtype+'clim_'+savenameext_nok+'.pickle','wb')
    pickle.dump([clim, trend], f_datac)
    f_datac.close()

# Scale by sqrt(cos(latitude))
data_scaled = np.zeros((nt,nl))
for i in range(nt):
    data_scaled[i,:] = data[i,:]*np.sqrt(np.cos(latipf*pi/180.))    

# Open EOF
if makeeof == 1 and eofnum > 0:
    print 'Making EOF...'
    lam, evectors, PC, PCmap_full, PCmap_anom = princomp(data_scaled, eofnum, removemean=0)
    lam_frac, lam_var_rep = analyse_evals(lam, len(lam))
    # Save EOF
    f_eof = open(datadir_out+'eof_'+savenameext_nok+'.pickle','wb')
    pickle.dump([lam,evectors,PC,PCmap_full,PCmap_anom], f_eof)
    f_eof.close()
else:
    print 'Loading EOF'
    f_eof = open(datadir_out+'eof_'+savenameext_nok+'.pickle','rb')
    lam,evectors,PC,PCmap_full,PCmap_anom = pickle.load(f_eof)
    lam_frac, lam_var_rep = analyse_evals(lam, len(lam))
    f_eof.close()

# Reconstruct data from EOF
PCmap_rec_full, PCmap_rec = eof_reconstruct(PCmap_anom, latipf, trend, clim)

if eofnum > 0:
    if clusteranom == 1:
        dataf = np.copy(PCmap_anom)
    else:
        dataf = np.copy(PCmap_rec)
else: 
    dataf = np.copy(data_scaled)

if saveproc:
    # Save processed data
    f_data = open(datadir_out+'dataf_'+savenameext_nok+'.pickle','wb')
    pickle.dump(dataf, f_data)
    f_data.close()

## Testing ##############
testing = 0
if testing:
    varrat = np.zeros(100)
    for i in range(100):
        varrat[i] = variance_ratio(PC[i,:])
    r_idx = argsort(varrat)
    varrat_r = varrat[r_idx]

    PC_r = PC[r_idx,:]
    evectors_r = evectors[:,r_idx]
    lam_frac_r = lam_frac[r_idx]
    lam_var_rep_r = np.zeros(100)
    for i in range(100):
        lam_var_rep_r[i] = sum(lam_frac_r[0:i+1])

    for i in range(30):
        print i+1, '\t', r_idx[i]+1, '\t', varrat[i], '\t', lam_frac[i], '\t', lam_var_rep[i], '\t'
    for i in range(30):
        print i+1, '\t', varrat_r[i], '\t', lam_frac_r[i], '\t', lam_var_rep_r[i], '\t'

    # Try keeping top 11 by variance ratio ranking. (varrat > 0.8)
    numpc_r = 11
    PCmap_anom_r = dot(evectors_r[:,range(numpc_r)], PC[range(numpc_r),:]).T
     
    PCmap_rec_full_r, PCmap_rec_r = eof_reconstruct(PCmap_anom_r, latipf, trend, clim)

    if eofnum > 0:
        if clusteranom == 1:
            dataf_r = np.copy(PCmap_anom_r)
        else:
            dataf_r = np.copy(PCmap_rec_r)
    else: 
        dataf_r = np.copy(data_scaled)

    saveproc=1
    if saveproc:
        # Save processed data
        f_data = open(datadir_out+'dataf_varianceratio_'+str(numpc_r)+'eof.pickle','wb')
        pickle.dump(dataf, f_data)
        f_data.close()
    

