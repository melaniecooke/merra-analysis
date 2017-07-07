#!/usr/bin/env python

import sys
import os
import numpy as np
sys.path.append('/home/mcooke/Research/KMeans/')
from numpy import zeros
from numpy import ones
from numpy import squeeze as sq
from pylab import *
import matplotlib.pyplot as plt
from scipy import interpolate
from math import *
import pickle
import mpl_toolkits.basemap as bm
import scipy.io
from met_utils import *
from netCDF4 import Dataset
import netCDF4 as netcdf4
from scipy.cluster.vq import *
from scipy.interpolate import griddata
from datetime import datetime as datetime
from calendar import monthrange
from scipy import stats
import time
import copy 
#
from kmeans_functions import *
from merra_process import *
from kmeans_merra_classif import *
from merra_atmos_plotsc import *
#from plot_tools_2 import *
from merra_opendap_tools import *

blue_red1, anomaly1, lightmidrb = pt.customcmaps()


def reconstructanom(A,clim):
    B = A/np.sqrt(np.cos(latipf*pi/180.))
    B = B + clim
    return B

## Initialize: User input ##

k = 7
itera = 50
clusteranom = 1
anomtype = 'd' # Use s for seasonal, d for daily

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

datadir_in = '/home/mcooke/Research/KMeans/'
datadir = '/home/mcooke/Research/KMeans/NewAlgorithm/NewMerra/ProcessedData/'
#kmeansdir = '/home/mcooke/Research/KMeans/NewAlgorithm/NewMerra/ProcessedData/multi_kmeans/'
eventdir = '/home/mcooke/Research/Algorithm_Testing/Algorithm_New/'

imgfolder = str(k) + '_' + str(itera) + '/'
figdir = '/home/mcooke/Research/KMeans/NewAlgorithm/NewMerra/Figures/K'+str(k)+'_'+str(itera)+'/'
savenameext = 'multik_'+str(k)+'_'+str(itera)
fext = '.pdf'
mydpi = 600

print 'Doing analysis for ', savenameext

##  Load data  ##
f_o = open(datadir+'dataf_H500_6-8_1979-'+str(yearf)+'_nosm_daily_18eof_d.pickle','rb')
data_o = pickle.load(f_o)
f_o.close()

f_c = open(datadir+'dclim_H500_6-8_1979-'+str(yearf)+'_nosm_daily_18eof_d.pickle','rb')
dclim, trend = pickle.load(f_c)
f_c.close()
augsummerind = 30+31+31

# Toolik Lake coordinates
latt = 68 + 38/60
#lont = 360 - 149 - 36/60
lont = -149 - 36/60

session, cred = merra_session()
f = merra_url(1999,7,1,datatype='slv')
with esgf_pydap.Dataset(f[0], session=session, **cred) as nc:
    lono = nc.variables['XDim'][:]
    lat = nc.variables['YDim'][:]

#lon = shiftwest(lono,lono)
lon = np.copy(lono)

for i in range(len(lon)):
    if lon[i] < 0:
        lon[i] = 360 + lon[i]

# Limiting grid
# (a,d)  (b,d)
# (a,c)  (b,c)
ao = 360-240 # 240 west
bo = 360-60  # 60 west
co = 35
do = 90
cro = 360-149.5
a = find_nearest_index(lon,ao)
b = find_nearest_index(lon,bo)
c = find_nearest_index(lat,co)
d = find_nearest_index(lat,do)
cr = find_nearest_index(lon,cro)

#lonsub = range(a,b+1) #for west-shifted lon
lonsub_e = range(a,len(lon))
lonsub_w = range(0,b+1)
latsub = range(c,d+1)

#loni = lon[lonsub] # for west-shifted lon
loni = np.concatenate((lon[lonsub_e],lon[lonsub_w]))
lati = lat[latsub]
loncr = lon[cr] # 149.375
nnx = len(loni)
nny = len(lati)
nn = nnx*nny
nz = 42
#nlx = len(loni[::4])
#nly = len(lati[::4])

nt, nlo = np.shape(data_o)

lonlr = loni[::4]
latlr = lati[::4]
lonip, latip = meshgrid(lonlr,latlr)
nly, nlx = np.shape(lonip)
nl = nly*nlx
lonipf = np.reshape(lonip, nl)
latipf = np.reshape(latip, nl)

m = bm.Basemap(width=7500000, height=5300000,
    resolution='l', projection='laea',\
    lat_ts=latt-3,lat_0=latt-3,lon_0=lont-2)
[x,y] = m(lonip,latip)
[xt, yt] = m(lont,latt)

fnamesave = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-2013.pickle'
fnamesave_time = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-2013.pickle'
fnamesave2 = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_2014-'+str(yearf)+'.pickle'
fnamesave2_time = datadir_in + 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_2014-'+str(yearf)+'.pickle'

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

## Open summer mean climatology ##
fh = datadir+'MERRA.climo.slv.H500.6-8_1979-'+str(yearf)+'.pickle'
with open(fh, 'rb') as f:
    clim = pickle.load(f)
climlr = clim[::4,::4]
climlr_flat = np.reshape(climlr,nl)

##  Open centroids  ##
fsavec = open(datadir+'multik_'+str(k)+'_'+str(itera)+'.pickle','rb')
CMo, CM, CS, patcor_mean = pickle.load(fsavec)
fsavec.close()

CM_Rec = CM + climlr_flat

IDX, dist2 = vq(data_o, CMo)
idx = list(IDX)

data_o_a = reconstructanom(data_o, np.zeros(np.shape(data_o)))
data_o_rec = reconstructanom(data_o, dclim)

# Counting # Fixed on September 6 2013
# And ordering based on distribution
countso = np.zeros(k)
for i in range(k):
    countso[i] = idx.count(i)
cind = np.argsort(countso)
cind = cind[::-1]
counts = countso[cind]
centroids = CM_Rec[cind,:]
centroids_a = CM[cind,:]
patcor = patcor_mean[cind]

# Categorize dates # Fixed on Sep 6 2013
idx_ordered = np.zeros(nt)
for i in range(nt):
    #idx_ordered[i] = cind[idx[i]] # Old method -- wrong!
    idx_ordered[i]  = np.where(cind==idx[i])[0][0]

with open(datadir+'km_ordered_centroids.pickle','wb') as f_cent:
    pickle.dump([centroids, centroids_a, patcor, idx_ordered, counts],f_cent)


#
### Plotting #######
#
C = np.reshape(centroids, (k, nly, nlx))
Ca = np.reshape(centroids_a, (k, nly, nlx))

rc('text', fontsize=14)

vals = range(520,610+4,4)
valsa = np.arange(-15,15+1,1)

### Individual plots of clusters
for i in range(k):
    #fig = plot_map(m,x,y,C[i,:,:]*0.1,vals,cm.Spectral_r)
    fig = plot_map2(1,m,x,y,Ca[i,:,:]*0.1,C[i,:,:]*0.1,valsa,vals,'hPa','hPa',lightmidrb)
    title('K = '+str(k)+', dist = '+str(round(dist2[i],2))+', Group '+str(i+1)+', n = '+str(counts[i]))
    fimg = figdir+'individualKmeans/km_'+savenameext+'_group_'+intto2dstr(i+1)+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.4)

# Grid plot of clusters (anomaly)
rc('text', fontsize=12)
#grtitle = ['Group %g, mean(patcor) = %.2f \n n = %g' % (pi+1, p, counts[pi]) for pi,p in enumerate(patcor)] 
fig = plot_grid2(1,m,x,y,k,0.1*C,Ca*0.1,vals,valsa,'decameters','decameters',0,lightmidrb,pltextrema=0,titles=1,plottoolik=0)
fig.set_size_inches(7.3,5.5)
#suptitle('500 hPa height centroids (dam), K = '+str(k))
fimg = figdir+'centroids.png'
dimg = ensure_dir(fimg)
plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.05)
fimg = figdir+'centroids.pdf'
plt.savefig(fimg, dpi=300, bbox_inches='tight', pad_inches=0.05)

# L&O format plot of anomaly centroids
#grtitle_lo = ['Regime %g' % (pi+1) for pi in range(k)]
#fig = plot_grid2(1,m,x,y,k,0.1*C,Ca*0.1,vals,valsa,'dam',grtitle_lo,lightmidrb) 
#        #if len(B[s]) > 0:
#fimg = figdir+'km_'+savenameext+'_grid_LOformat'+fext
#dimg = ensure_dir(fimg)
#plt.savefig(fimg, dpi=600, bbox_inches='tight', pad_inches=0.1)

# Grid plot of clusters (full)
rc('text', fontsize=9)
#grtitle_wl = ['%g \n n_mix = %g' % (pi+1, counts[pi]) for pi,p in enumerate(patcor)] 
grtitle = ['Group %g, mean(patcor) = %.2f \n n = %g' % (pi+1, p, counts[pi]) for pi,p in enumerate(patcor)] 
fig = plot_grid(2,m,x,y,k,0.1*C,vals,'dam',grtitle,lightmidrb,cline=1)
suptitle('500 hPa heights centroids (dam), K = '+str(k))
fimg = figdir+'km_'+savenameext+'_grid_full.png'
dimg = ensure_dir(fimg)
plt.savefig(fimg, dpi=150, bbox_inches='tight', pad_inches=0.15)


print 'End.'
