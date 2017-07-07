#!/usr/bin/env python

import sys
import os
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import squeeze as sq
#import matplotlib # for use without X forwarding
#matplotlib.use('Agg') # for use without X forwarding
from pylab import *
import matplotlib.pyplot as plt
from scipy import interpolate
from math import *
import pickle
import mpl_toolkits.basemap as bm
import scipy.io
from met_utils import *
import plot_tools as pt
from netCDF4 import Dataset
import netCDF4 as netcdf4
from scipy.cluster.vq import *
from scipy.interpolate import griddata
from calendar import monthrange
from scipy import stats
import time
import copy 

startstart = time.time()

## Initialize: User input ##
make = 0
makeeof = 0
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
k = 6
yeari = 1979
yearf = 2013
yearlist = range(yeari,yearf+1)
yeari2 = 1999
yearf2 = 2010
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

print 'Doing analysis for ', savenameext

## 1. Initialize: definitions ##
directory = {}
filebase = {}
types = []
directory['slv'] = 'http://goldsmr2.sci.gsfc.nasa.gov/opendap/MERRA/MAT1NXSLV.5.2.0/'
filebase['slv'] = '.prod.assim.tavg1_2d_slv_Nx.'
types.append('slv')

Mversion = ['MERRA100','MERRA200','MERRA300', 'MERRA301']

fileext = '.hdf'

blue_red1, anomaly1, lightmidrb = pt.customcmaps()

# General
def filename(year,month,day,vnum,datatype):
    #if type(day) is int:
    #    day = intto2dstr(day)
    fdir = directory[datatype] + str(year) + '/' + intto2dstr(month) + '/'
    f = Mversion[vnum] + filebase[datatype] + str(year) + intto2dstr(month) + intto2dstr(day) + fileext
    return fdir + f

def ensure_dir(f):
    import errno
    d = os.path.dirname(f)
    #if not os.path.exists(d):
    #    os.makedirs(d)
    try:
        os.makedirs(d)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return d

def shiftwest(A,lono):
    # A is field to be shifted, lono is original unshifted longitude grid associated with A
    midp = findclosest1d(lono,0)
    Awest = A[0:midp]
    Aeast = A[midp::]
    Ashifted = np.concatenate((Aeast,Awest))
    return Ashifted

def gridcluster(k,ny,nx,centroids):
    C = np.zeros((k,ny,nx))
    nn = ny*nx
    for i in range(k):
        #cent = nan*np.ones(nn)
        #cent[ninds] = centroids[i,:]
        cent = centroids[i,:]
        C[i,:,:] = np.reshape(cent,(ny,nx))
    return C

def princomp(A, numpc=0, removemean=1): 
    # Modified from The Glowing Python http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html
    """ performs principal components analysis (PCA) on the n-by-p data matrix A.
    p Rows of A correspond to observations , n columns to variables. 
    S-mode (EOF mode):     domain p = Rows = space, variables n = Columns = time : (time x space) input (unrotated) 
    T-mode (Circ pattern): domain p = Rows = time, variables n = Columns = space : (space x time) input 
    
    Returns :  
     evectors :
       is a p-by-p matrix, each column containing coefficients 
       for one principal component.
     PC : 
       the principal component scores; that is, the representation 
       of A in the principal component space. Rows of SCORE 
       correspond to observations, columns to components.
     lam : 
       a vector containing the eigenvalues 
       of the covariance matrix of A.
    """
    from numpy import mean,cov,cumsum,dot,linalg,size,flipud
    # computing eigenvalues and eigenvectors of covariance matrix
    n,p = np.shape(A)
    if removemean == 1:
        M = (A - np.mean(A.T, axis=1)).T # subtract the mean (along columns)
        #                                # and transpose. M (p x n)
    else:
        M = A.T
    cM = cov(M) # Covariance matrix of M. cM (p x p)
    [lam,evectors] = linalg.eig(cM) # Eigenvectors of the covariance matrix
    #                               # lam (p x 1)               
    #                               # evectors (p x p)
    p = size(evectors,axis=1)
    idx = argsort(lam)  # sorting the eigenvalues
    idx = idx[::-1]     # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    evectors = evectors[:,idx].real # Corresponds to the maps of each PC
    lam = lam[idx].real # sorting eigenvalues
    #if numpc < p or numpc >= 0:
    #    evectors = evectors[:,range(numpc)] # cutting some PCs
    PC = dot(evectors.T, M) # projection of the data in the new space
    #                       # PC (numpc x n)
    #                       # these are the timeseries
    PCmap_full = dot(evectors[:,range(numpc)], PC[range(numpc),:]).T + mean(A, axis=0) # Reconstruction of original data from numpc evectors (n x p)
    PCmap_anom = dot(evectors[:,range(numpc)], PC[range(numpc),:]).T # Reconstruction of anomalies (n x p)
    return lam, evectors, PC, PCmap_full, PCmap_anom

def analyse_evals(lam,numpc):
    lam_frac = 100*(lam[0:numpc]/sum(lam))
    lam_var_rep = np.zeros(numpc)
    for i in range(numpc):
        lam_var_rep[i] = sum(lam_frac[0:i+1])
    return lam_frac, lam_var_rep

# Classifiability
def spatialanom(p):
    pp = np.zeros(len(p))
    pmean = np.mean(p)
    for i in range(len(p)):
        pp[i] = p[i] - pmean
    return pp

def patcor(p,q):
    pp = spatialanom(p)
    qp = spatialanom(q)
    numerator = np.sum(pp*qp)
    denom = np.sqrt(np.sum(pp**2)*np.sum(qp**2))
    return numerator/denom

def makeA(P,Q):
    k, M = np.shape(P)
    A = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            A[i,j] = patcor(P[i],Q[j])
    return A

def minvalA(A):
    return np.nanmin(np.nanmin(A))

def classindex(itera,ALLP):
    minsum = 0
    for m in range(itera):
        for mp in range(itera):
            if m != mp:
                A = makeA(ALLP[m],ALLP[mp])
                minsum = minsum + minvalA(A)
    return (1.0/(itera*(itera-1)))*minsum

# Plotting tools
def plot_map(m,x,y,A,vals,colormap,fignum=1):
    #
    fig = plt.figure(fignum)
    clf()
    #adjustFigAspect(fig, aspect=1)
    m.drawcoastlines(color=[0.13,0.23,0.17], linewidth=0.5)
    m.drawmeridians(arange(0,360,30),labels=[0,0,0,1])
    m.drawparallels(arange(20,90,15),labels=[1,0,0,0])
    #vals = range(460,600,4)
    cs2 = m.contour(x, y, A, vals, extend='both', linewidths=0.5, inline=1, colors='k')
    plt.clabel(cs2, fmt='%1.0f', fontsize=8, inline=1)
    #vals2 = range(520,610+4,4)
    cs1 = m.contourf(x, y, A, vals, extend='both', cmap=colormap)
    cbh = colorbar(cs1)
    return fig

def plot_map2(m,x,y,A,B,vals,vals2,colormap,fignum=1):
    #
    fig = plt.figure(fignum)
    clf()
    #adjustFigAspect(fig, aspect=1)
    m.drawcoastlines(color=[0.13,0.23,0.17], linewidth=0.5)
    m.drawmeridians(arange(0,360,30),labels=[0,0,0,1])
    m.drawparallels(arange(20,90,15),labels=[1,0,0,0])
    #vals = range(460,600,4)
    #vals2 = range(520,610+4,4)
    cs1 = m.contourf(x, y, A, vals, extend='both', cmap=colormap)
    cbh = colorbar(cs1)
    cbh.set_label('Anomaly (dam)')
    cs2 = m.contour(x, y, B, vals2, extend='both', linewidths=0.5, inline=1, colors='k')
    plt.clabel(cs2, fmt='%1.0f', fontsize=8, inline=1)
    return fig

def plot_anom_grid(fignum,m,x,y,k,field,climo,vals,vals2,units,counts,colormap,cline=1):
    #ptnum = 2
    #clm = [30, 20, 15, 20, 20]
    #units = ["mb", "C", "dam", "C", "C"]
    #vals = [range(958,1066,3), range(-40,40,4), range(460,600,4), range(-40,40,4), range(-30,40,4)]
    #contour_levels = clm[ptnum]*arange(-1,1.1,0.1)
    fig = plt.figure(fignum)
    clf()
    k,_,_ = np.shape(field)
    for s in range(k):
        if k > 6:
            subplot(3,3,s+1)
        else:
            subplot(3,2,s+1)
        m.drawcoastlines(color=[0.35,0.25,0.35], linewidth=0.25)
        cs = m.contourf(x, y, field[s,...]-climo, vals2, cmap=colormap, extend='both')
        cs1 = m.contour(x, y, field[s,...], vals, colors='k', linewidths=0.5, inline=1, extend='both')
        plt.clabel(cs1, cs1.levels[::2], fmt='%1.0f', fontsize=6, inline=1)
        title('Group '+str(s+1)+', n = '+str(counts[s]))
    subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = colorbar(cs, cax=cbar_ax)
    cbar.set_label('Anomaly ('+units+')')
    return fig

def plot_grid2(fignum,m,x,y,k,field,field2,vals,vals2,cbarl,counts,colormap):
    fig = plt.figure(fignum)
    clf()
    for s in range(k):
        if k > 6:
            subplot(3,3,s+1)
        else:
            subplot(3,2,s+1)
        m.drawcoastlines(color=[0.35,0.25,0.35], linewidth=0.25)
        cs = m.contourf(x, y, field2[s,...], vals2, cmap=colormap, extend='both')
        cs1 = m.contour(x, y, field[s,...], vals, colors='k', linewidths=0.5, inline=1, extend='both')
        plt.clabel(cs1, cs1.levels[::2], fmt='%1.0f', fontsize=6, inline=1)
        title('Group '+str(s+1)+', n = '+str(counts[s]))
    subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = colorbar(cs, cax=cbar_ax)
    cbar.set_label(cbarl)
    return fig

def plot_grid(fignum,m,x,y,k,field,vals,units,counts,colormap,cline=1):
    fig = plt.figure(fignum)
    clf()
    for s in range(k):
        if k > 6:
            subplot(3,3,s+1)
        else:
            subplot(3,2,s+1)
        m.drawcoastlines(color=[0.35,0.25,0.35], linewidth=0.25)
        cs = m.contourf(x, y, field[s,::], vals, cmap=colormap, extend='both')
        if cline == 1:
            cs1 = m.contour(x, y, field[s,::], vals, colors='k', linewidths=0.5, inline=1, extend='both')
            plt.clabel(cs1, cs1.levels[::2], fmt='%1.0f', fontsize=6, inline=1)
        title('Group '+str(s+1)+', n = '+str(counts[(s)]))
    subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = colorbar(cs, cax=cbar_ax)
    cbar.set_label(units)
    return fig

## Find closest cluster given a specific date (datetime)
#def closestcluster(d, centroids, varname, Mversion):
#    hourgrid = range(0,24+3,3)
#    hrind = findclosest1d(hourgrid,d.hour+d.minute/60.)
#    # Need flag for when hour is closer to 24
#    if hourgrid[hrind] == 24:
#        d = d + timedelta(1)
#        hrind = 0
#    for vnum in range(len(Mversion)):
#        try:
#            f = filename(d.year,d.month,d.day,vnum,'slv')
#            with Dataset(f) as nc:
#                Al = sq(nc.variables[varname][hrind,latsub,lonsub_e])
#                Ar = sq(nc.variables[varname][hrind,latsub,lonsub_w])
#                #t = np.shape(H500l)[0]
#                #times = nc.variables['TIME'][t/2]
#                #timeun = nc.variables['TIME'].units
#            break
#        except RuntimeError as err:
#            if vnum <= 2:
#                pass
#            else:
#                print 'No valid file found for ', year, month, day
#                raise
#    A = np.concatenate((Al,Ar),axis=2)

def closestcluster(year, eventno, datatype, varname, centroids, k, avg24=0, anom=0):
    # TO DO: Add anomaly option (make sure it's an equivalent anomaly to the centroid's)
    # Add 24 hour averaging option
    timeunits2 = 'days since 1999-06-01 00:00:00'
    di = datetime(1999,6,1,12)
    df = datetime(1999,8,31,12)
    daygrid = np.arange(netcdf4.date2num(di, timeunits2, calendar='standard'),netcdf4.date2num(df, timeunits2, calendar='standard')+1,1)
    k, nl = np.shape(centroids)
    f = "/home/mcooke/Research/MERRA_cuts/" + datatype + "/MERRA." + datatype + ".cuts." + str(year) + "." + intto2dstr(eventno) + ".nc"
    with Dataset(f) as nc:
        if avg24:
            A = nc.variables['H500'][2:5+1,::]
            A = np.mean(A, axis=0)
        else:
            A = nc.variables['H500'][4,::]
        timeunits = nc.variables['TIME'].__getattribute__('units')
        t = nc.variables['TIME'][4]
        da = netcdf4.num2date(t, timeunits, calendar='standard')
        da2 = datetime(1999,da.month,da.day,da.hour,da.minute)
    A = A[::4,::4]
    Af = np.reshape(A, nl)
    if anom == 1:
        dd = netcdf4.date2num(da2, timeunits2, calendar='standard')
        if dd > 100: print 'Warning: day outside range'
        dayindex = findclosest1d(daygrid, dd)
        Af = Af - dclimlr[dayindex,:] 
    distances = np.zeros(k)
    for ki in range(k):
        d = 0
        for xi in range(nl):
            d += (Af[xi] - centroids[ki,xi])**2
        distances[ki] = d
    kimin = argmin(distances)
    return kimin, distances, A, Af, da

    


## 2. Initialize: assign variables ##
hlevind = [0, 3, 6, 12, 16] # 1000, 925, 850, 700, 500
hind1000 = 0
hind975 = 1
hind850 = 2
hind700 = 3
hind500 = 4

f = filename(1999,7,1,1,'slv')
with Dataset(f) as nc:
    lono,lat = nc.variables['XDim'][:], nc.variables['YDim'][:]

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
a = findclosest1d(lon,ao)
b = findclosest1d(lon,bo)
c = findclosest1d(lat,co)
d = findclosest1d(lat,do)

#lonsub = range(a,b+1) #for west-shifted lon
lonsub_e = range(a,len(lon))
lonsub_w = range(0,b+1)
latsub = range(c,d+1)

#loni = lon[lonsub] # for west-shifted lon
loni = np.concatenate((lon[lonsub_e],lon[lonsub_w]))
lati = lat[latsub]
nnx = len(loni)
nny = len(lati)
nn = nnx*nny

## 3. Load data ##

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
if make==1:
    for year in yearlist:
        for month in monthlist: 
            monthlen = monthrange(year,month)[1]
            print 'Loading Year', year, ', Month', month
            for day in range(1,monthlen+1):
                for vnum in range(len(Mversion)):
                    try:
                        f = filename(year,month,day,vnum,'slv')
                        with Dataset(f) as nc:
                            H500l = sq(nc.variables[varname][:,latsub,lonsub_e])
                            H500r = sq(nc.variables[varname][:,latsub,lonsub_w])
                            t = np.shape(H500l)[0]
                            times = nc.variables['TIME'][t/2]
                            timeun = nc.variables['TIME'].units
                        break
                    except RuntimeError as err:
                        if vnum <= 2:
                            pass
                        else:
                            print 'No valid file found for ', year, month, day
                            raise
                H500 = np.concatenate((H500l,H500r),axis=2)
                datets = netcdf4.num2date(times, timeun, calendar='standard')
                # Average over one day (one loaded file)
                H500A[iter1,:,:] = np.mean(H500, axis=0)
                data[iter1,:] = np.reshape(H500A[iter1,:,:], nn)
                alltimes[iter1] = netcdf4.date2num(datets, 'days since 1800-01-01 00:00:00', calendar='standard')
                iter1+=1
    print 'Data loaded:'
    print time.time() - start
    if varname == 'H500':
        fnamesave = 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
        fnamesave2 = 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'
    else:
        fnamesave = 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
        fnamesave2 = 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'
    fsave = open(fnamesave,'wb')
    pickle.dump(H500A,fsave)
    fsave.close()
    fsave = open(fnamesave2,'wb')
    pickle.dump(alltimes,fsave)
    fsave.close()
else:
    if varname == 'H500':
        fnamesave = 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
        fnamesave2 = 'merra_slv_daily_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'
    else:
        fnamesave = 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_'+str(yeari)+'-'+str(yearf)+'.pickle'
        fnamesave2 = 'merra_slv_daily_'+varname+'_'+str(monthi)+'-'+str(monthf)+'_times_'+str(yeari)+'-'+str(yearf)+'.pickle'
    fsave = open(fnamesave,'rb')
    H500A = pickle.load(fsave)
    fsave.close()
    t = np.shape(H500A)[0]
    for ti in range(t):
        data[ti,:] = np.reshape(H500A[ti,:,:], nn)
    fsave = open(fnamesave2,'rb')
    alltimes = pickle.load(fsave)
    fsave.close()

# Load climo
fh = '/home/mcooke/Research/Climo/MERRA.climo.slv.' + savenameexts + '.nc'
with Dataset(fh) as nc:
    Hc = sq(nc.variables[varname][:])


## 4a. Process data ## 

# Adjust data to be low res
H500Alr = H500A[:,::4,::4]
lonlr = loni[::4]
latlr = lati[::4]
Hclr = Hc[::4,::4]

# Reshape
nly, nlx = np.shape(H500Alr)[1::]
nl = nlx*nly
datalr = np.zeros((nt,nl))
for i in range(nt):
    datalr[i,:] = np.reshape(H500Alr[i,:,:], nl)

datalro = np.copy(datalr)
Hcflat = np.reshape(Hclr,nl)

lonip, latip = meshgrid(lonlr,latlr)
lonipf = np.reshape(lonip,nl)
latipf = np.reshape(latip,nl)

lonip, latip = meshgrid(lonlr,latlr)

# Toolik Lake coordinates
latt = 68 + 38/60
#lont = 360 - 149 - 36/60
lont = -149 - 36/60

# Map base for plotting
blue_red1, anomaly1, lightmidrb = pt.customcmaps()

m = bm.Basemap(width=7500000, height=5300000,
    resolution='l', projection='laea',\
    lat_ts=latt-3,lat_0=latt-3,lon_0=lont-2)
[x,y] = m(lonip,latip)
[xt, yt] = m(lont,latt)

# Make daily climatology
yearlen = len(yearlist)
slen = nt/yearlen
mlen = len(monthlist)
# Data reshaped with years in individual rows (#years,#days,#gridpoints)
data_st = np.reshape(datalr, (len(yearlist),nt/len(yearlist),nl))
dclimlr = np.mean(data_st, axis=0)
dclimlr_long = np.zeros((nt,nl))
for i in range(yearlen):
    dclimlr_long[i*slen:(i+1)*slen,:] = np.copy(dclimlr)
# Make seasonal climatology
sclimlr = {}
mfinish = 0
for mi in range(mlen):
    monthlen = monthrange(2003,monthlist[mi])[1]   
    mstart = mfinish
    mfinish = mstart+monthlen
    sclimlr[str(mi)] = np.mean(np.mean(data_st[:,mstart:mfinish,:], axis=1),axis=0)
sclimlr_yr = np.zeros((slen,nl))
mfinish = 0
for mi in range(mlen):
    monthlen = monthrange(2003,monthlist[mi])[1]
    mstart = mfinish
    mfinish = mstart+monthlen
    sclimlr_yr[mstart:mfinish,:] = np.ones((monthlen,nl))*sclimlr[str(mi)]
sclimlr_long = np.zeros((nt,nl))
for i in range(yearlen):
    sclimlr_long[i*slen:(i+1)*slen,:] = np.copy(sclimlr_yr)
# Make anomalies
data_sta = data_st - dclimlr
datalra = np.reshape(data_sta, (nt,nl))
datalras = datalr - sclimlr_long
# Define test coordinates
flatcoords_150W_80N = 1496+34
flatcoords_150W_50N = 476+34

# Make test plot of daily climatology
climtestplot=0
if climtestplot == 1:
    fig = plt.figure(9)
    clf()
    ax = subplot(2,1,1)
    ax.set_color_cycle([cm.rainbow(mm) for mm in linspace(0,1,yearlen)])
    plot(range(slen),data_st[:,:,flatcoords_150W_80N].T)
    plot(range(slen),dclimlr[:,flatcoords_150W_80N], color='k',linewidth=3)
    xlim(0,slen)
    title('150W, 80N')
    ax = subplot(2,1,2)
    ax.set_color_cycle([cm.rainbow(mm) for mm in linspace(0,1,yearlen)])
    plot(range(slen),data_st[:,:,flatcoords_150W_50N].T)
    plot(range(slen),dclimlr[:,flatcoords_150W_50N], color='k',linewidth=3)
    xlim(0,slen)
    title('150W, 50N')
    suptitle('Daily climo test')
    fimg = 'Figures/climtestplot.png'
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

# Make detrended data
print 'Detrending...'
datalra_detr = np.zeros((nt,nl))
datalras_detr = np.zeros((nt,nl))
trend = np.zeros((nt,nl))
for n in range(nl):
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(nt),datalra[:,n])
    for i in range(nt):
        trend[i,n] = slope*i + intercept
    datalra_detr[:,n] = datalra[:,n] - trend[:,n]
    datalras_detr[:,n] = datalras[:,n] - trend[:,n]

# Make test plot of detrended data
detrtestplot=0
if detrtestplot == 1:
    fig = plt.figure(8)
    clf()
    for i, si in enumerate([flatcoords_150W_80N,flatcoords_150W_50N]):
        ax = subplot(2,1,i+1)
        plot(datalra[:,si], color=[.8,.8,.8])
        plot(datalra_detr[:,si], color='k')
        plot(trend[:,si], color=[.8,0,0])
        xlim(0,nt)
    fimg = 'Figures/detrtest.png'
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
# Make test plot of detrended data
detrtestplot=0
if detrtestplot == 1:
    fig = plt.figure(8)
    clf()
    for i, si in enumerate([flatcoords_150W_80N,flatcoords_150W_50N]):
        ax = subplot(2,1,i+1)
        plot(datalras[:,si], color=[.8,.8,.8])
        plot(datalras_detr[:,si], color='k')
        plot(trend[:,si], color=[.8,0,0])
        xlim(0,nt)
    fimg = 'Figures/detrtest_s.png'
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

#Plot trend map
plottrendmap=0
if plottrendmap:
    trendmap = np.reshape(trend[-1,:]-trend[0,:],(nly,nlx))
    fig = plot_map(m,x,y,0.1*trendmap,30,cm.YlOrRd)
    title('Linear trend in MERRA JJA data 1979-2013 (dam)')
    fimg = 'Figures/trend.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if anomtype == 's':
    data_a_d = np.copy(datalras_detr)
elif anomtype == 'd':
    data_a_d = np.copy(datalra_detr)

if clusteranom == 1:
    datalr = np.copy(data_a_d)
elif anomtype == 's':
    datalr = data_a_d + sclimlr_long
elif anomtype == 'd':
    datalr = data_a_d + dclimlr_long

# Do low pass filter on data without running years together
if timefilter > 0:
    r = timefilter+1
    #
    yearinds = np.where(diff(alltimes) > 1)[0]
    #
    dataf = np.zeros((nt,nl))
    for yy in range(len(yearinds)):
        if yy == 0:
            yi = 0
            yf = yearinds[0]+1
            for i in range(nl):
                dataf[yi:yf, i] = smooth(datalr[yi:yf, i], r)
        elif yy == len(yearinds)-1:
            yi = yearinds[yy-1]+1
            for i in range(nl):
                dataf[yi::, i] = smooth(datalr[yi::, i], r)
        else:
            yi = yearinds[yy-1]+1
            yf = yearinds[yy]+1
            for i in range(nl):
                dataf[yi:yf, i] = smooth(datalr[yi:yf, i], r)
else:
    dataf = np.copy(datalr)

# Apply day skip if relevant
if dayskip > 0:
    dataf = dataf[::dayskip,:]
    alltimes2 = alltimes[::dayskip]
else:
    alltimes2 = np.copy(alltimes)

# Scale by sqrt(cos(latitude))
data_scaled = np.zeros((nt,nl))
for i in range(nt):
    data_scaled[i,:] = dataf[i,:]*np.sqrt(np.cos(latipf*pi/180.))    

ploteigenvalues=1
plotEOFs = 1
#Apply PCA filtering
if makeeof == 1 and eofnum > 0:
    print 'Making EOF...'
    lam, evectors, PC, PCmap_full, PCmap_anom = princomp(data_scaled, eofnum, removemean=0)
    lam_frac, lam_var_rep = analyse_evals(lam, len(lam))
    # Save EOF
    f_eof = open('eof_'+savenameext_nok+'.pickle','wb')
    pickle.dump([lam,evectors,PC,PCmap_full,PCmap_anom], f_eof)
    f_eof.close()
    #
    print 'eofnum =', eofnum, ',', lam_var_rep[eofnum], '% variability represented'
    lneval = np.log(lam)
    dlneval = np.abs(np.diff(lneval))
    ## Reconstruct full field
    #rescaled = PCmap_anom/np.sqrt(np.cos(latipf*pi/180.))
    #retrend = rescaled + trend
    ##retrend = np.reshape(retrend, (yearlen,slen,nl))
    ##PCmap_rec = retrend + dclimlr
    ##PCmap_rec = np.reshape(PCmap_rec, (nt,nl))
    #PCmap_rec = retrend + dclimlr_long
    # Plot eigenvalue vs component
    if ploteigenvalues == 1:
        fig = plt.figure(12)
        clf()
        # lam_frac
        subplot(3,1,1)
        plot(range(1,nl+1),lam_frac, color='k')
        xlim(1,50)
        xticks(range(5,50,5))
        grid(True)
        title('Fraction of variance represented by each component')
        xlabel('PC component number')
        # lam_var_rep (cumulative)
        subplot(3,1,2)
        plot(range(1,nl+1),lam_var_rep, color='k')
        xlim(1,50)
       # xticks(range(5,50,5))
        grid(True)
        title('Cumulative variance represented')
        xlabel('PC component number')
        # component-component change in ln(eigenvalue)
        subplot(3,1,3)
        plot(range(1,nl),dlneval, color='k')
        xlim(1,50)
        xticks(range(5,50,5))
        grid(True)
        title('Component-component change in ln(eigenvalue)')
        xlabel('PC component number')
        suptitle('Determination of PC cutoff')
        fimg = 'Figures/pc_cutoff_'+savename_nok+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
elif makeeof == 0 and eofnum > 0:
    # Open EOF
    print 'Loading EOF'
    f_eof = open('eof_'+savenameext_nok+'.pickle','rb')
    lam,evectors,PC,PCmap_full,PCmap_anom = pickle.load(f_eof)
    lam_frac, lam_var_rep = analyse_evals(lam, len(lam))
    f_eof.close()

plotEOFs = 1
if plotEOFs == 1 and eofnum > 0:
    EOF = evectors.T/np.sqrt(np.cos(latipf*pi/180.))
    EOF = np.reshape(EOF, (nl, nly, nlx))
    EOFfull = np.zeros((nl, nl))
    for i in range(nl):
        EOFfull[i,:] = (sqrt(lam[i])*evectors[:,i].T)/np.sqrt(np.cos(latipf*pi/180.)) + np.mean(datalr, axis=0)
    EOFfull = np.reshape(EOFfull, (nl, nly, nlx))
    eofvals = 0.1*np.arange(-80,80+5,5)
    vals = range(520,610+4,4)
    eofvals2 = vals
    for n in range(6):
        fig = plot_map2(m,x,y,0.1*sqrt(lam[n])*EOF[n,:,:],0.1*EOFfull[n,:,:],eofvals,eofvals2,cm.RdBu_r)
        title('Detrended, weighted EOF '+str(n+1)+', '+str(round(lam_frac[n],2))+'% variability represented')
        #title('Unrotated detrended PC component '+str(n+1))
        fimg = 'Figures/'+imgfolder+'EOF_'+savenameext_nok+'_'+intto2dstr(n+1)+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if eofnum > 0:
    # Reconstruct full field
    rescaled = PCmap_anom/np.sqrt(np.cos(latipf*pi/180.))
    retrend = rescaled + trend
    if anomtype == 'd':
        PCmap_rec_full = retrend + dclimlr_long
        PCmap_rec = PCmap_anom + dclimlr_long
    elif anomtype == 's':
        PCmap_rec_full = retrend + sclimlr_long
        PCmap_rec = PCmap_anom + sclimlr_long
    if clusteranom == 1:
        dataf = np.copy(PCmap_anom)
    else:
        dataf = np.copy(PCmap_rec)
else:
    dataf = np.copy(data_scaled)

## 4b. Run cluster analysis

nindst = np.isfinite(dataf[:,0])
#ninds = np.isfinite(dataf[0,:]) # Merra data has no nans in the domain we use

data2 = dataf[nindst,:] # Make new continuous data set without nans
alltimes2 = alltimes2[nindst]
nt2 = np.shape(data2)[0]
#k = 7 
if makecent == 1:
    start = time.time()
    # Use kmeans. Runs algorithm iter times and returns centroids with lowest distortion seen.
    # Kmeans algorithm starts from randomly generated selections as centroids each time
    # Takes about 20 min with iter=20
    #centroids, dist = kmeans(data2, k, iter=20)
    centroids, dist = kmeans(data2, k, iter=20)
    print 'Centroids computed'
    print (time.time() - start)
    #
    fsavec = open('merra_slv_centroids_'+savenameext+'.pickle','wb')
    pickle.dump([centroids,dist],fsavec)
    fsavec.close()
else:
    fsavec = open('merra_slv_centroids_'+savenameext+'.pickle','rb')
    centroids, dist = pickle.load(fsavec)
    fsavec.close()

idx, dist2 = vq(data2, centroids)
idx = list(idx)

# Counting # Fixed on September 6 2013
centroidso = np.copy(centroids)
countso = np.zeros(k)
for i in range(k):
    countso[i] = idx.count(i)
cind = np.argsort(countso)
cind = cind[::-1]
counts = countso[cind]
centroids = centroidso[cind]

# Categorize dates # Fixed on Sep 6 2013
idx_ordered = np.zeros(nt2)
for i in range(nt2):
    #idx_ordered[i] = cind[idx[i]] # Old method -- wrong!
    idx_ordered[i]  = np.where(cind==idx[i])[0][0]

# Make reconstructed clusters from anomalies
if clusteranom == 1:
    centroids_a = np.copy(centroids)
    # Reconstruct full field
    crescaled = centroids_a/np.sqrt(np.cos(latipf*pi/180.))
    #retrend = rescaled + trend
    centroids_rec_full = crescaled + Hcflat
    centroids_rec = centroids_a + Hcflat
    centroids = np.copy(centroids_rec_full)

def reconstructanom(A,clim):
    B = A/np.sqrt(np.cos(latipf*pi/180.))
    B = B + clim
    return B
    

# Calculate duration histogram
durn = []
d = 0
durncat = {}
for ki in range(k): durncat[str(ki)] = []
for i in range(1,len(idx)):
    if idx[i] == idx[i-1]:
        d+=1 # increment d
    else:
        durn.append(d)
        durncat[str(int(idx[i-1]))].append(d)
        d = 0

# Transfer matrix
transf = np.zeros((k,k))
for i in range(len(idx)-1):
    cur = idx[i]
    next = idx[i+1]
    transf[cur,next]+=1

if plotdurn == 1:
    fig = plt.figure(15)
    clf()
    #n,bins,patches = plt.hist(durn, 20, color=[.6,.6,.6])
    #binlen = round(diff(bins)[0])
    #binrange = 20*binlen
    #bins2 = np.arange(0,binrange+binlen,binlen)
    #clf()
    bins2 = concatenate(([0,1],range(3,33,3),[60]))
    n,bins,patches = plt.hist(durn, bins2, color=[.6,.6,.6])
    xticks(bins2)
    xlabel('Cluster duration (days)')
    title('Persistence of clusters')
    fimg = 'Figures/'+imgfolder+'persistence_hist_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    fig = plt.figure(16)
    for ki in range(k):
        clf()
        n,bins,patches = plt.hist(durncat[str(ki)], bins2, color=[.6,.6,.6])
        xticks(bins2)
        ylim(0,70)
        xlabel('Cluster duration (days)')
        title('Persistence of clusters')
        fimg = 'Figures/'+imgfolder+'persistence_hist_'+savenameext+'_'+str(ki)+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

plottransf = 1
t_mask = np.zeros((k,k))
for ki in range(k): t_mask[ki,ki] = 1
t_mask2 = np.ones((k,k))
for ki in range(k): t_mask2[ki,ki] = 0
if plottransf:
    fig = plt.figure(17)
    clf()
    tx, ty = np.meshgrid(range(1,k+2),range(1,k+2))
    subplot(2,1,1, axisbg=[0.8,0.8,0.8])
    pch = pcolor(tx, ty, np.ma.array(transf, mask=t_mask2), cmap=cm.gist_earth_r)
    cbh = colorbar(pch)
    xticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    yticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    ylabel('Cluster group')
    subplot(2,1,2, axisbg=[0.8,0.8,0.8])
    pch = pcolor(tx, ty, np.ma.array(transf, mask=t_mask), cmap=cm.gist_earth_r)
    cbh = colorbar(pch)
    xticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    yticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    ylabel('Cluster group')
    xlabel('Next cluster group')
    fimg = 'Figures/'+imgfolder+'transf_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    
    

## 5. Comparison with event dates ##
f_events = open('../events_AAS.pickle', 'rb')
f_events_t = open('../events_AAStype.pickle', 'rb')
A = pickle.load(f_events)
At = pickle.load(f_events_t)
#yeari2 = 1999
#yearf2 = 2011
#yearlist2 = range(yeari2,yearf2+1)

if clusteranom:
    data2_rec = reconstructanom(data2,dclimlr_long)
    data2_a = reconstructanom(data2,np.zeros(np.shape(data2)))
else:
    data2_rec = reconstructanom(data2,np.zeros(np.shape(data2)))
    data2_a = data2_rec - dclimlr_long

A_k = {}
A_k_flat = []
B = {}
M = {}
Y = {}
ToD = {}
MY = np.zeros((k,len(yearlist2),len(monthlist)))
A_k2 = {}
A_k_flat2 = []
B2 = {}
M2 = {}
Y2 = {}
ToD2 = {}
MY2 = np.zeros((k,len(yearlist2),len(monthlist)))
lake_d_i = []
Aamap_all = np.zeros((k,nl))
Afmap_all = np.zeros((k,nl))
Oamap_all = np.zeros((k,nl))
Ofmap_all = np.zeros((k,nl))
for ki in range(k):
    B[str(ki)] = np.array([], dtype=object)
    M[str(ki)] = []
    Y[str(ki)] = []
    ToD[str(ki)] = []
    B2[str(ki)] = np.array([], dtype=object)
    M2[str(ki)] = []
    Y2[str(ki)] = []
    ToD2[str(ki)] = []
for year in yearlist2:
    A_k[str(year)] = nan*np.ones(len(A[str(year)]))
    A_k2[str(year)] = nan*np.ones(len(A[str(year)]))
    # Cycle through events
    for j in range(len(A[str(year)])):
        # Original method
        dt = timedelta(A[str(year)][j] - 1)
        d = datetime(year,1,1,0,0,0) + dt
        dt_timezone = timedelta(hours=8)
        d = d + dt_timezone # Adjust Alaska time zone (AKDT) to UTC
        #
        if d.month in monthlist:
            # Original method:
            #df = datetime(d.year, d.month, d.day, 12, 30)
            df = datetime(d.year, d.month, d.day, d.hour, d.minute)
            dfn = netcdf4.date2num(df, 'days since 1800-01-01 00:00:00', calendar='standard')
            d_i = findclosest1d(alltimes2,dfn)
            A_k[str(year)][j] = idx_ordered[d_i]
            A_k_flat.append(idx_ordered[d_i])
            lake_d_i.append(d_i)
            # Add to composite
            ki = int(idx_ordered[d_i])
            Oamap_all[ki,:] = Oamap_all[ki,:] + data2_a[d_i,:]
            Ofmap_all[ki,:] = Ofmap_all[ki,:] + data2_rec[d_i,:]
            # New method:
            if clusteranom == 1:
                ki, distances, Afmap, Aamap, da = closestcluster(year, j, 'slv', 'H500', crescaled, k, avg24=1, anom=1)
            else:
                ki, distances, Afmap, Aamap, da = closestcluster(year, j, 'slv', 'H500', centroids, k, avg24=1)
            A_k2[str(year)][j] = ki
            A_k_flat2.append(ki)
            # Add to composite
            Aamap_all[ki,:] = Aamap_all[ki,:] + Aamap
            Afmap_all[ki,:] = Afmap_all[ki,:] + np.reshape(Afmap, nl)
            #
            # List of event dates grouped by kmeans cluster membership
            # Old method:
            id = str(int(idx_ordered[d_i]))
            # B: [event id, doy from event algorithm, datetime (minute accuracy), closest daily date]
            if len(B[id]) < 1:
                B[id] = np.array([[[year,j], A[str(year)][j], df, alltimes2[d_i]]])
            else:
                B[id] = np.concatenate((B[id], np.array([[[year,j], A[str(year)][j], df, alltimes2[d_i]]])), axis=0)
            M[id].append(d.month)
            Y[id].append(d.year)
            # New method:
            id2 = str(int(ki))
            if len(B2[id2]) < 1:
                B2[id2] = np.array([[[year,j], A[str(year)][j], da, alltimes2[d_i]]])
            else:
                B2[id2] = np.concatenate((B2[id2], np.array([[[year,j], A[str(year)][j], da, alltimes2[d_i]]])), axis=0)
            M2[id2].append(da.month)
            Y2[id2].append(da.year)
            #
            # Find time of day
            # Old method:
            dtime = d - dt_timezone # Go back to AKDT
            timeofday = dtime.hour + dtime.minute*(1/60.)
            ToD[id].append(timeofday) # AKDT
            MY[int(id),d.year-yeari2,d.month-monthlist[0]]+=1
            # New method:
            dtime = da - dt_timezone # Go back to AKDT
            timeofday = dtime.hour + dtime.minute*(1/60.)
            ToD2[id2].append(timeofday) # AKDT
            MY2[int(id2),da.year-yeari2,da.month-monthlist[0]]+=1

# Average Aamap_all (anomaly map for clustered anomalies)
# and Afmap_all (full map in both cases - not flattened)
# and Oamap_all and Ofmap_all (for index method)
for ki in range(k):
    Afmap_all[ki,:] = Afmap_all[ki,:]/float(A_k_flat2.count(ki))
    Aamap_all[ki,:] = Aamap_all[ki,:]/float(A_k_flat2.count(ki))
for ki in range(k):
    Ofmap_all[ki,:] = Ofmap_all[ki,:]/float(A_k_flat.count(ki))
    Oamap_all[ki,:] = Oamap_all[ki,:]/float(A_k_flat.count(ki))

# Check lake event recategorization
lerecat = np.zeros((k,k))
for year in yearlist2:
    for j in range(len(A[str(year)])):
        cur = A_k[str(year)][j]
        next = A_k2[str(year)][j]
        if isfinite(cur) and isfinite(next):
            lerecat[int(cur),int(next)]+=1

plotlerecat = 0
ler_mask = np.zeros((k,k))
for ki in range(k): ler_mask[ki,ki] = 1
ler_mask2 = np.ones((k,k))
for ki in range(k): ler_mask2[ki,ki] = 0
if plotlerecat:
    fig = plt.figure(18)
    clf()
    tx, ty = np.meshgrid(range(1,k+2),range(1,k+2))
    subplot(2,1,1, axisbg=[0.8,0.8,0.8])
    pch = pcolor(tx, ty, np.ma.array(lerecat, mask=ler_mask), cmap=cm.gist_earth_r)
    cbh = colorbar(pch)
    xticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    yticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    ylabel('Index grouping')
    subplot(2,1,2, axisbg=[0.8,0.8,0.8])
    pch = pcolor(tx, ty, np.ma.array(lerecat, mask=ler_mask2), cmap=cm.gist_earth_r)
    cbh = colorbar(pch)
    xticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    yticks(np.arange(1.5,k+1.5,1), [str(xii) for xii in range(1,k+1)])
    ylabel('Index grouping')
    xlabel('Closest centroid grouping')
    fimg = 'Figures/'+imgfolder+'lerecat_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

#f_B = open('events_km_'+folderext+'_'+savenameext+'.pickle','wb')
f_B = open('events_km_'+savenameext+'.pickle','wb')
pickle.dump([B,A_k], f_B)
f_B.close()

f_dist = open('distribution_lakev_km_'+savenameext+'.pickle','wb')
pickle.dump([M,Y,ToD], f_dist)
f_dist.close()

#f_B = open('events_km_'+folderext+'_'+savenameext+'_cl.pickle','wb')
f_B = open('events_km_'+savenameext+'_cl.pickle','wb')
pickle.dump([B2,A_k2], f_B)
f_B.close()

f_dist = open('distribution_lakev_km_'+savenameext+'_cl.pickle','wb')
pickle.dump([M2,Y2,ToD2], f_dist)
f_dist.close()

#Create distribution statistics for full clusters
Ma = {}
Ya = {}
ToDa = {}
for ki in range(k):
    id = str(ki)
    Ma[id] = []
    Ya[id] = []
    ToDa[id] = []
for i in range(nt2):
    da = netcdf4.num2date(alltimes2[i], 'days since 1800-01-01 00:00:00', calendar='standard')
    id = str(int(idx_ordered[i]))
    Ma[id].append(da.month)
    Ya[id].append(da.year)
    ToDa[id].append(da.hour) # UTC

f_dist = open('distribution_all_km_'+savenameext+'.pickle','wb')
pickle.dump([Ma,Ya,ToDa], f_dist)
f_dist.close()

# Save non-lake event dates
alltimes2_nl = np.delete(alltimes2, lake_d_i)
idx_ordered_nl = np.delete(idx_ordered, lake_d_i)
alltimes_sorted_nl = {}
for ki in range(k):
    id = str(ki)
    alltimes_sorted_nl[id] = alltimes2_nl[idx_ordered_nl == ki] # days since 1800-01-01 00:00:00

f_C = open('events_km_all_'+folderext+'_'+savenameext+'.pickle','wb')
pickle.dump(alltimes_sorted_nl, f_C)
f_C.close()

#
def countevents(k,catlist,M):
    Cm = np.zeros((k+1,len(catlist)))
    for i, ci in enumerate(catlist):
        for ki in range(k):
            Cm[ki,i] = M[str(ki)].count(ci)
            Cm[k,i] = Cm[k,i] + Cm[ki,i]
    Cmo = np.copy(Cm)
    for ki in range(k+1):
        Cm[ki,:] = 100*(Cm[ki,:]/np.sum(Cm[ki,:]))
    return Cm, Cmo

def countevents_time(k, ToD):
    timelist = np.arange(0,24+1,3)
    timelistmid = timelist[0:-1] + 1.5
    Ct = np.zeros((k+1,len(timelistmid)))
    for ki in range(k):
        ntod, bins = np.histogram(ToD[str(ki)],timelist, normed=False)
        Ct[ki,:] = ntod
        Ct[k,:] = Ct[k,:] + Ct[ki,:]
    Cto = np.copy(Ct)
    for ki in range(k+1):
        Ct[ki,:] = 100*(Ct[ki,:]/np.sum(Ct[ki,:]))
    return Ct, Cto
 
    
## Counting
# Old:
counts_e = np.zeros(k)
for i in range(k):
    counts_e[i] = A_k_flat.count(i)

Cev = np.array((counts,counts_e))
for n in range(2):
    Cev[n,:] = 100*(Cev[n,:]/np.sum(Cev[n,:]))
# New:
counts_e2 = np.zeros(k)
for i in range(k):
    counts_e2[i] = A_k_flat2.count(i)

Cev2 = np.array((counts,counts_e2))
for n in range(2):
    Cev2[n,:] = 100*(Cev2[n,:]/np.sum(Cev2[n,:]))

# Distribution - Lake events only
# Old method
Cm, Cmo = countevents(k, monthlist, M)

# New method
Cm2, Cmo2 = countevents(k, monthlist, M2)
    
# Old method
Cy, Cyo = countevents(k, yearlist2, Y)

# New method
Cy2, Cyo2 = countevents(k, yearlist2, Y2)

# Old method
timelist = np.arange(0,24+1,3)
timelistmid = timelist[0:-1] + 1.5
Ct, Cto = countevents_time(k, ToD)
# New method
Ct2, Cto2 = countevents_time(k, ToD2)

Cm_avg = np.zeros((k+1,2))
Cy_avg = np.zeros((k+1,2))
for ki in range(k+1):
    Cm_avg[ki,0] = np.mean(Cm[ki,:])
    Cm_avg[ki,1] = np.std(Cm[ki,:])

# Distribution - All

CAm, CAmo = countevents(k, monthlist, Ma)
CAy, CAyo = countevents(k, yearlist, Ya)


CAm_avg = np.zeros((k+1,2))
CAy_avg = np.zeros((k+1,2))
for ki in range(k+1):
    CAm_avg[ki,0] = np.mean(CAm[ki,:])
    CAm_avg[ki,1] = np.std(CAm[ki,:])


## 6. Plotting ##


C = gridcluster(k,nly,nlx,centroids)
if clusteranom:
    Ca = gridcluster(k,nly,nlx,crescaled)

rc('text', fontsize=10)

def dist_plot(fignum, k, nlist, xtype, xlist, l_legname, C):
    fig = plt.figure(fignum)
    clf()
    ax = gca()
    ax.set_color_cycle([cm.Set1(mm) for mm in linspace(0,1,10)])
    ccycle = [cm.Set1(mm) for mm in linspace(0,1,10)]
    ccycle[k] = [0.4,0.4,0.4]
    width = 1./len(nlist) - 0.025
    widthtot = width*len(nlist)
    barlocs = np.array(range(len(xlist))) - 0.5*widthtot
    ph = []
    for n in nlist:
        h = bar(barlocs, C[n,:], width, color=ccycle[n])
        ph.append(h)
        barlocs = barlocs+width
    xlabel(xtype)
    xlim(-1, len(xlist))
    xticks(range(len(xlist)),xlist)
    ylabel('Amount in '+xtype+' (%)')
    legh = legend(ph, l_legname, loc=0)
    ltext = legh.get_texts()
    setp(ltext, fontsize=7)
    return fig 

def dist_plot2(fignum, k, nlist, xtype, xlist, l_legname, C):
    fig = plt.figure(fignum)
    clf()
    ax = gca()
    ax.set_color_cycle([cm.Set1(mm) for mm in linspace(0,1,10)])
    ccycle = [cm.Set1(mm) for mm in linspace(0,1,10)]
    ccycle[k] = [0.4,0.4,0.4]
    width = 1./len(nlist) - 0.025
    widthtot = width*len(nlist)
    barlocs = np.array(range(len(xlist))) - 0.5*widthtot
    ph = []
    for n in nlist:
        h = bar(barlocs, C[n,:], width, color=ccycle[n])
        ph.append(h)
        barlocs = barlocs+width
    xlabel(xtype)
    xlim(-1, len(xlist))
    xticks(range(len(xlist)),xlist)
    ylabel('Amount in '+xtype)
    legh = legend(ph, l_legname, loc=0)
    ltext = legh.get_texts()
    setp(ltext, fontsize=7)
    return fig 


vals = range(520,610+4,4)
valsa = np.arange(-15,15+1,1)
if clusterplot == 1:
    ### Individual plots of clusters
    vals = range(520,610+4,4)
    #for i in range(k):
    #    fig = plot_map(m,x,y,C[i,:,:]*0.1,vals,cm.Spectral_r)
    #    title('K = '+str(k)+', dist = '+str(round(dist,2))+', Group '+str(i+1)+', n = '+str(counts[i]))
    #    fimg = 'Figures/'+imgfolder+'new/indv/km_'+savenameext+'_group_'+intto2dstr(i+1)+'.png'
    #    dimg = ensure_dir(fimg)
    #    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.4)

    # Grid plot of clusters (anomaly)
    if not clusteranom:
        fig = plot_anom_grid(2,m,x,y,k,C*0.1,Hclr*0.1,vals,np.arange(-15,15+1,1),'dam',counts,lightmidrb,cline=1)
    else:
        fig = plot_grid2(2,m,x,y,k,C*0.1,Ca*0.1,vals,np.arange(-15,15+1,1),'dam',counts,lightmidrb)
    #fignum,m,x,y,field,climo,vals,vals2,units,counts,colormap,cline=1
    #suptitle('K = '+str(k)+', dist = '+str(round(dist,2)))
    suptitle('Clustered 500 hPa heights (dam), K = '+str(k))
    fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_grid.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    # Grid plot of clusters
    fig = plot_grid(10,m,x,y,k,0.1*C,vals,'dam',counts,lightmidrb,cline=1)
    suptitle('Clustered 500 hPa heights (dam), K = '+str(k))
    fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_grid_full.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    #
    checkclosestplot = 1
    if clusteranom and checkclosestplot:
        #fig = plot_grid2(11,m,x,y,k,0.1*Ca,0.1*np.reshape(Afmap_all-crescaled,(k,nly,nlx)),np.arange(-15,15+1,1),np.arange(-15,15+1,1),'dam',counts,lightmidrb)
        fig = plot_grid2(11,m,x,y,k,0.1*Ca,0.1*np.reshape(Aamap_all,(k,nly,nlx)),np.arange(-15,15+1,1),np.arange(-15,15+1,1),'dam',counts_e2,lightmidrb)
        suptitle('Clustered 500 hPa height anomalies (dam, contours) and closest-grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_cla.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plot_grid2(11,m,x,y,k,0.1*C,0.1*np.reshape(Afmap_all,(k,nly,nlx)),vals,vals,'dam',counts_e2,lightmidrb)
        suptitle('Clustered 500 hPa heights (dam, contours) and closest-grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_clf.png'
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plot_grid2(11,m,x,y,k,0.1*Ca,0.1*np.reshape(Oamap_all,(k,nly,nlx)),valsa,valsa,'dam',counts_e,lightmidrb)
        suptitle('Clustered 500 hPa height anomalies (dam, contours) and index grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_oa.png'
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plot_grid2(11,m,x,y,k,0.1*C,0.1*np.reshape(Ofmap_all,(k,nly,nlx)),vals,vals,'dam',counts_e,lightmidrb)
        suptitle('Clustered 500 hPa heights (dam, contours) and index grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_of.png'
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plot_grid2(11,m,x,y,k,0.1*Ca,0.1*np.reshape(Oamap_all-Aamap_all,(k,nly,nlx)),valsa,valsa,'dam',counts_e,lightmidrb)
        suptitle('Clustered 500 hPa height anomalies (dam, contours) and index-closest grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_diff_a.png'
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plot_grid2(11,m,x,y,k,0.1*C,0.1*np.reshape(Ofmap_all-Afmap_all,(k,nly,nlx)),vals,valsa,'dam',counts_e,lightmidrb)
        suptitle('Clustered 500 hPa heights (dam, contours) and index-closest grouped lake events (shading), K = '+str(k))
        fimg = 'Figures/'+imgfolder+'km_'+savenameext+'_checkclosest_diff_f.png'
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

distpctplot = 1
alldistplot = 1
distplotcl = 1
distplotind = 0
if distplot == 1:
    rc('text', fontsize=12)
    # Distribution of events in clusters
    fig = plt.figure(3)
    clf()
    ax = gca()
    ax.set_color_cycle([cm.Set1(mm) for mm in linspace(0,1,9)])
    ccycle = [cm.Set1(mm) for mm in linspace(0,1,9)]
    width = 0.45
    widthtot = width*2
    barlocs = np.array(range(k)) - 0.5*widthtot
    ph = []
    for n in range(2):
        h = bar(barlocs, Cev[n,:], width, color=ccycle[n])
        ph.append(h)
        barlocs = barlocs+width
    xlabel('K')
    xlim(-1, k)
    xticks(range(k),range(1,k+1),fontsize=10)
    ylabel('Amount in group (%)')
    l_legname = ['All data','Events']
    legh = legend(ph, l_legname, loc=0)
    ltext = legh.get_texts()
    setp(ltext, fontsize=14)
    fimg = 'Figures/'+imgfolder+'events_in_daily_kmeans_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=100, bbox_inches='tight', pad_inches=0.35)
    
    # Distribution of events in clusters
    fig = plt.figure(3)
    clf()
    ax = gca()
    ax.set_color_cycle([cm.Set1(mm) for mm in linspace(0,1,9)])
    ccycle = [cm.Set1(mm) for mm in linspace(0,1,9)]
    width = 0.45
    widthtot = width*2
    barlocs = np.array(range(k)) - 0.5*widthtot
    ph = []
    for n in range(2):
        h = bar(barlocs, Cev2[n,:], width, color=ccycle[n])
        ph.append(h)
        barlocs = barlocs+width
    xlabel('K')
    xlim(-1, k)
    xticks(range(k),range(1,k+1),fontsize=10)
    ylabel('Amount in group (%)')
    l_legname = ['All data','Events']
    legh = legend(ph, l_legname, loc=0)
    ltext = legh.get_texts()
    setp(ltext, fontsize=14)
    fimg = 'Figures/'+imgfolder+'events_in_daily_kmeans_'+savenameext+'_closest.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=100, bbox_inches='tight', pad_inches=0.35)

if distplot and distplotind:
    rc('text', fontsize=10)
    # Distribution of events - months and years
    #l_legname = ['k=1','k=2','k=3','k=4','k=5', 'All k']
    l_legname_o = ['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','All k']
    l_legname = l_legname_o[0:k] + [l_legname_o[-1]]
    klist = range(0,k+1)
    klist2 = range(0,k)
    #
    if distpctplot == 1:
        fig = dist_plot(31, k, klist2, 'month',  monthlist, l_legname, Cm)
        title('Distribution of lake events by month')
        fimg = 'Figures/'+imgfolder+'event_distribution_month_'+savenameext+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

        fig = dist_plot(31, k, klist2, 'year', yearlist2, l_legname, Cy)
        title('Distribution of lake events by year')
        fimg = 'Figures/'+imgfolder+'event_distribution_year_'+savenameext+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    fig = dist_plot2(31, k, klist, 'month', monthlist, l_legname, Cmo)
    title('Distribution of lake events by month')
    fimg = 'Figures/'+imgfolder+'event_distribution_month_'+savenameext+'_fullamt.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    fig = dist_plot2(31, k, klist, 'year', yearlist2, l_legname, Cyo)
    title('Distribution of lake events by year')
    fimg = 'Figures/'+imgfolder+'event_distribution_year_'+savenameext+'_fullamt.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    # Time of day
    fig = dist_plot(3, k, klist, 'time of day', timelistmid, l_legname, Ct)
    title('Distribution of lake events by time of day')
    fimg = 'Figures/'+imgfolder+'event_distribution_ToD_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if distplot and distplotcl:
    rc('text', fontsize=10)
    # Distribution of events - months and years
    #l_legname = ['k=1','k=2','k=3','k=4','k=5', 'All k']
    l_legname_o = ['k=1','k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','All k']
    l_legname = l_legname_o[0:k] + [l_legname_o[-1]]
    klist = range(0,k+1)
    klist2 = range(0,k)
    #
    if distpctplot == 1:
        fig = dist_plot(31, k, klist2, 'month',  monthlist, l_legname, Cm2)
        title('Distribution of lake events by month')
        fimg = 'Figures/'+imgfolder+'event_distribution_month_'+savenameext+'_closest.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

        fig = dist_plot(31, k, klist2, 'year', yearlist2, l_legname, Cy2)
        title('Distribution of lake events by year')
        fimg = 'Figures/'+imgfolder+'event_distribution_year_'+savenameext+'_closest.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    fig = dist_plot2(31, k, klist, 'month', monthlist, l_legname, Cmo2)
    title('Distribution of lake events by month')
    fimg = 'Figures/'+imgfolder+'event_distribution_month_'+savenameext+'_fullamt_closest.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    fig = dist_plot2(31, k, klist, 'year', yearlist2, l_legname, Cyo2)
    title('Distribution of lake events by year')
    fimg = 'Figures/'+imgfolder+'event_distribution_year_'+savenameext+'_fullamt_closest.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    # Time of day
    fig = dist_plot(3, k, klist, 'time of day', timelistmid, l_legname, Ct2)
    title('Distribution of lake events by time of day')
    fimg = 'Figures/'+imgfolder+'event_distribution_ToD_'+savenameext+'_closest.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if distplot and alldistplot:
    # All days distribution
    # year
    if distpctplot == 1:
        fig = dist_plot(31, k, klist, 'year', yearlist, l_legname, CAy)
        title('Distribution of clustered days by year')
        fimg = 'Figures/'+imgfolder+'all_distribution_year_'+savenameext+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        # month
        fig = dist_plot(31, k, klist, 'month', monthlist, l_legname, CAm)
        title('Distribution of clustered days by month')
        fimg = 'Figures/'+imgfolder+'all_distribution_month_'+savenameext+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    #
    fig = dist_plot2(31, k, klist, 'year', yearlist, l_legname, CAyo)
    title('Distribution of clustered days by year')
    fimg = 'Figures/'+imgfolder+'all_distribution_year_'+savenameext+'_fullamt.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    #
    fig = dist_plot2(31, k, klist, 'month', monthlist, l_legname, CAmo)
    title('Distribution of clustered days by month')
    fimg = 'Figures/'+imgfolder+'all_distribution_month_'+savenameext+'_fullamt.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if plot_boxplot == 1:
    # Average number of events per year and per month, with spread
    fig = plt.figure(4)
    clf()
    h = boxplot(np.reshape(MY,(k,len(yearlist)*len(monthlist))).T)
    xlabel('k')
    #xlim(-1, k+1)
    #xticks(range(k+1), ['1','2','3','4','all'])
    ylim(-2,8)
    ylabel('Number of events per month')
    title('Monthly event statistics')
    fimg = 'Figures/'+imgfolder+'event_distribution_month_boxplot_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    fig = plt.figure(4)
    clf()
    h = boxplot(np.sum(MY,2).T)
    xlabel('k')
    #xlim(-1, k+1)
    #xticks(range(k+1), ['1','2','3','4','all'])
    ylim(-2,12)
    ylabel('Number of events per year')
    title('Yearly event statistics')
    fimg = 'Figures/'+imgfolder+'event_distribution_year_boxplot_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)


## 7. Classifiability ##
if makeclass == 1:
    nnc = np.shape(data2)[1]
    krange = range(2,10+1)
    CL = np.zeros(len(krange))
    start = time.time()
    #ALLP = {}
    #ALLdist = {}
    #ALLidx = {}
    classifiability = {}
    for j, k in enumerate(krange):
        print k
        # Use kmeans. Runs algorithm iter times and returns centroids with lowest distortion seen.
        # Kmeans algorithm starts from randomly generated selections as centroids each time
        #ALLP{str(k)} 
        itera = 50
        #ALLP[str(k)] = np.zeros((itera,k,nnc))
        #ALLdist[str(k)] = np.zeros(itera)
        #ALLidx[str(k)] = np.zeros((itera,nt))
        ALLP = np.zeros((itera,k,nnc))
        ALLdist = np.zeros(itera)
        ALLidx = np.zeros((itera,nt2))
        for mm in range(itera):
            centroidsc, distc = kmeans(data2, k, iter=5)
            idxc, dist2c = vq(data2, centroidsc)
            #idxc = list(idxc)
            #ALLP[str(k)][mm,:,:] = centroidsc
            #ALLdist[str(k)][mm] = distc
            #ALLidx[str(k)][mm,:] = idxc
            ALLP[mm,:,:] = centroidsc
            ALLdist[mm] = distc
            ALLidx[mm,:] = idxc
        #classifiability[str(k)] = classindex(itera,ALLP[str(k)])
        classifiability[str(k)] = classindex(itera,ALLP)
        CL[j] = classifiability[str(k)]
    print (time.time() - start)

    f_class = open('classifiability_'+str(itera)+'_km_'+savenameext+'.pickle','wb')
    pickle.dump([ALLP,itera,CL], f_class)
    f_class.close()

    fig = plt.figure(4)
    clf()
    plot(krange, CL, color='k')
    xlabel('K')
    ylabel('Classifiability index')
    fimg = 'Figures/'+imgfolder+'new/classifiability50_'+savenameext+'.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

## 8. EOF ##
if makepc == 1:
    numpc = 12
    lam, evectors, PC, PCmap_full, PCmap_anom = princomp(dataf, numpc)
    lam_frac, lam_var_rep = analyse_evals(lam, numpc)
    #
    print 'numpc =', numpc, ',', lam_var_rep[-1], '% variability represented'
    #
    EOF = np.reshape(evectors.T, (numpc, nly, nlx))
    EOFfull = np.zeros((numpc, nl))
    for i in range(numpc):
        EOFfull[i,:] = sqrt(lam[i])*evectors[:,i].T + np.mean(dataf, axis=0)
    EOFfull = np.reshape(EOFfull, (numpc, nly, nlx))
    eofvals = 0.1*np.arange(-80,80+5,5)
    vals = range(520,610+4,4)
    eofvals2 = vals
    for n in range(numpc/2):
        fig = plot_map2(m,x,y,0.1*sqrt(lam[n])*EOF[n,:,:],0.1*EOFfull[n,:,:],eofvals,eofvals2,anomaly1)
        title('EOF '+str(n+1)+', '+str(round(lam_frac[n],2))+'% variability represented, '+savenameexts)
        fimg = 'Figures/'+imgfolder+'new/pc/EOF_'+intto2dstr(n+1)+'_'+savenameext_eof+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #
        fig = plt.figure(4)
        clf() 
        plot(PC[n,:], color='k')
        xlabel('time (days)')
        title('EOF '+str(n+1)+', '+str(round(lam_frac[n],2))+'% variability represented, '+savenameexts)
        fimg = 'Figures/'+imgfolder+'new/pc/timeseries_'+intto2dstr(n+1)+'_'+savenameext_eof+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

if pcatest == 1:

    # Rotated PC
    numpc2 = 30
    lam2, evectors2, PC2, PCmap_full2, PCmap_anom2 = princomp(datalr.T, numpc2)
    lam_frac2, lam_var_rep2 = analyse_evals(lam2, numpc2)
    #
    print 'numpc =', numpc2, ',', lam_var_rep2[-1], '% variability represented'
    #
    MAPmodes = np.reshape(PC2, (numpc2, nly, nlx))
    lneval = np.log(lam2)
    dlneval = np.abs(np.diff(lneval))
    # Plot eigenvalue vs component
    fig = plt.figure(10)
    clf()
    plot(range(1,len(dlneval)),dlneval, color='k')
    ylabel('Change in ln(eigenvalue)')
    xlabel('PC component number')
    xticks(range(5,50,5))
    xlim(1,50)
    title('Determination of PC cutoff')
    fimg = 'Figures/'+imgfolder+'new/pc/lneigenvalue-t.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    # Plot first few PCs
    for n in range(6):
        fig = plot_map(m,x,y,0.1*MAPmodes[n,:,:],np.arange(-900,930,30),cm.RdBu_r)
        title('T-mode PC component '+str(n+1))
        fimg = 'Figures/'+imgfolder+'new/pc/PC-t_'+savenameext_eof+'_'+intto2dstr(n+1)+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    # Detrend
    #from scipy import signal
    from scipy import stats
    yearlen = len(yearlist)
    slen = nt/yearlen
    data_st = np.reshape(datalr, (len(yearlist),nt/len(yearlist),nl))
    data_yrmean = np.mean(data_st, axis=1)
    data_yrmean_ext = np.zeros(np.shape(datalr))
    for i in range(yearlen):
        data_yrmean_ext[i*slen:(i+1)*slen,:] = np.ones((slen,nl))*data_yrmean[i,:]

    fig = plt.figure(11)
    clf()
    ni = 1
    for n in [500,1000,1500]:
        subplot(3,1,ni)
        #detrendtest = signal.detrend(datalr[:,n])
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(yearlist)),data_yrmean[:,n])
        trend = np.zeros(nt)
        for i in range(yearlen):
            trend[i*slen:(i+1)*slen] = np.ones(slen)*(slope*i)
        #trend = slope*np.arange(len(yearlist))
        #data_detr = data_yrmean[:,n] - trend
        data_detr = datalr[:,n] - trend
        plot(datalr[:,n],color=[.6,.6,.6])
        plot(data_yrmean_ext[:,n],color='r')
        plot(trend+intercept,color='b')
        xlim(0,nt-1)
        #plot(data_detr,color='k')
        title('n = '+str(n))
        ni+=1
    suptitle('Detrend test')
    fimg = 'Figures/'+imgfolder+'new/pc/detrendtest.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    data_detr = np.zeros((nt,nl))
    trend = np.zeros((nt,nl))
    for n in range(nl):
        #if n%200 == 0:
        #    print n
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(yearlist)),data_yrmean[:,n])
        #trend = np.zeros(nt)
        for i in range(yearlen):
            trend[i*slen:(i+1)*slen,n] = np.ones(slen)*(slope*i)
        #trend = slope*np.arange(len(yearlist))
        #data_detr = data_yrmean[:,n] - trend
        data_detr[:,n] = datalr[:,n] - trend[:,n]

    #Plot trend map
    trendmap = np.reshape(0.1*trend[slen,:],(nly,nlx))
    fig = plot_map(m,x,y,trendmap,30,cm.YlOrRd)
    title('Linear trend in MERRA JJA data 1979-2013 (dam/yr)')
    fimg = 'Figures/'+imgfolder+'new/pc/trend.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    # Unrotated detrended PC
    numpc3 = 30
    lam3, evectors3, PC3, PCmap_full3, PCmap_anom3 = princomp(data_detr, numpc3)
    lam_frac3, lam_var_rep3 = analyse_evals(lam3, numpc3)
    #
    print 'numpc =', numpc3, ',', lam_var_rep3[-1], '% variability represented'
    #
    lneval3 = np.log(lam3)
    dlneval3 = nnp.diff(lneval3)
    # Plot eigenvalue vs component
    fig = plt.figure(12)
    clf()
    plot(lam_frac3, color='k')
    ylabel('eigenvalue')
    xlabel('PC component number')
    title('Determination of PC cutoff')
    fimg = 'Figures/'+imgfolder+'new/pc/eigenvalue_detr.png'
    dimg = ensure_dir(fimg)
    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
    # Plot first few PCs
    EOF3 = np.reshape(evectors3.T, (numpc3, nly, nlx))
    EOFfull3 = np.zeros((numpc3, nl))
    for i in range(numpc3):
        EOFfull3[i,:] = sqrt(lam3[i])*evectors3[:,i].T + np.mean(data_detr, axis=0)
    EOFfull3 = np.reshape(EOFfull3, (numpc3, nly, nlx))
    eofvals = 0.1*np.arange(-80,80+5,5)
    vals = range(520,610+4,4)
    eofvals2 = vals
    for n in range(6):
        fig = plot_map2(m,x,y,0.1*sqrt(lam3[n])*EOF3[n,:,:],0.1*EOFfull3[n,:,:],eofvals,eofvals2,anomaly1)
        title('Detrended EOF '+str(n+1)+', '+str(round(lam_frac3[n],2))+'% variability represented, '+savenameexts)
        #title('Unrotated detrended PC component '+str(n+1))
        fimg = 'Figures/'+imgfolder+'new/pc/PC-s_detr_proj'+savenameext_eof+'_'+intto2dstr(n+1)+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    def subset_pc(A, evectors, numpc_old, numpc_new):
        if numpc_new < numpc_old:
            evectors = evectors[:,range(numpc_new)]
        #PC = PC[range(numpc_new),:]
        M = (A - np.mean(A.T, axis=1)).T
        PC = dot(evectors.T, M)
        PCmap_full = dot(evectors, PC).T + mean(A, axis=0)
        PCmap_anom = dot(evectors, PC).T
        return evectors, PC, PCmap_full, PCmap_anom

    #if makecent_t == 1:
    numpc3 = 12
    lam3, evectors3, PC3, PCmap_full3, PCmap_anom3 = princomp(data_detr, numpc3)
    lam_frac3, lam_var_rep3 = analyse_evals(lam3, numpc3)
    #
    print 'numpc =', numpc3, ',', lam_var_rep3[-1], '% variability represented'

    # Subset 
    numpc4 = 7
    evectors4, PC4, PCmap_full4, PCmap_anom4 = subset_pc(data_detr, evectors3, numpc3, numpc4)
    data_detr_pc = np.copy(PCmap_full4)
    numpc3 = numpc4
    k = 8
    start = time.time()
    # Use kmeans. Runs algorithm iter times and returns centroids with lowest distortion seen.
    # Kmeans algorithm starts from randomly generated selections as centroids each time
    # Takes about 20 min with iter=20
    #centroids, dist = kmeans(data2, k, iter=20)
    centroids, dist = kmeans(data_detr_pc, k, iter=20)
    print 'Centroids computed'
    print (time.time() - start)
    #
    fsavec = open('merra_slv_centroids_detr_pc'+str(numpc3)+'_k'+str(k)+'.pickle','wb')
    pickle.dump([centroids,dist],fsavec)
    fsavec.close()

    idx, dist2 = vq(data_detr_pc, centroids)
    idx = list(idx)

    # Counting # Fixed on September 6 2013
    centroidso = np.copy(centroids)
    countso = np.zeros(k)
    for i in range(k):
        countso[i] = idx.count(i)
    cind = np.argsort(countso)
    cind = cind[::-1]
    counts = countso[cind]
    centroids = centroidso[cind]

    # Categorize dates # Fixed on Sep 6 2013
    idx_ordered = np.zeros(nt2)
    for i in range(nt2):
        #idx_ordered[i] = cind[idx[i]] # Old method -- wrong!
        idx_ordered[i]  = np.where(cind==idx[i])[0][0]

    # Calculate duration histogram
    durn = []
    d = 0
    for i in range(1,len(idx)):
        if idx[i] == idx[i-1]:
            d+=1 # increment d
        else:
            durn.append(d)
            d = 0

    savenameext = 'H500_6-8_1979-2013_detr_pc'+str(numpc3)+'_daily_k'+str(k)
    if plotdurn == 1:
        fig = plt.figure(15)
        clf()
        #n,bins,patches = plt.hist(durn, 20, color=[.6,.6,.6])
        #binlen = round(diff(bins)[0])
        #binrange = 20*binlen
        #bins2 = np.arange(0,binrange+binlen,binlen)
        #clf()
        bins2 = concatenate(([0,1],range(3,33,3),[60]))
        n,bins,patches = plt.hist(durn, bins2, color=[.6,.6,.6])
        xticks(bins2)
        xlabel('Cluster duration (days)')
        title('Persistence of clusters: k = '+str(k)+', PCA cutoff = '+str(numpc3))
        fimg = 'Figures/'+imgfolder+'new/persistence_hist_'+savenameext+'.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

    C2 = gridcluster(k,nly,nlx,centroids)
    if clusterplot == 1:
        ### Individual plots of clusters
        vals = range(520,610+4,4)
        #for i in range(k):
        #    fig = plot_map(m,x,y,C[i,:,:]*0.1,vals,cm.Spectral_r)
        #    title('K = '+str(k)+', dist = '+str(round(dist,2))+', Group '+str(i+1)+', n = '+str(counts[i]))
        #    fimg = 'Figures/'+imgfolder+'new/indv/km_'+savenameext+'_group_'+intto2dstr(i+1)+'.png'
        #    dimg = ensure_dir(fimg)
        #    plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.4)

        # Grid plot of clusters (anomaly)
        fig = plot_anom_grid(2,m,x,y,k,C2*0.1,Hclr*0.1,vals,np.arange(-15,15+1,1),'dam',counts,anomaly1,cline=1)
        #fignum,m,x,y,field,climo,vals,vals2,units,counts,colormap,cline=1
        #suptitle('K = '+str(k)+', dist = '+str(round(dist,2)))
        suptitle('Clustered 500 hPa heights (dam), K = '+str(k)+', PCA cutoff = '+str(numpc3))
        fimg = 'Figures/'+imgfolder+'new/km_'+savenameext+'_grid.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)

        # Grid plot of clusters
        fig = plot_grid(10,m,x,y,k,0.1*C2,vals,'dam',counts,lightmidrb,cline=1)
        suptitle('Clustered 500 hPa heights (dam), K = '+str(k)+', PCA cutoff = '+str(numpc3))
        fimg = 'Figures/'+imgfolder+'new/km_'+savenameext+'_grid_full.png'
        dimg = ensure_dir(fimg)
        plt.savefig(fimg, dpi=120, bbox_inches='tight', pad_inches=0.35)
        #

print 'End.'
