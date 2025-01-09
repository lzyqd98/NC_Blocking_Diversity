#%%
###### This code is to study the blocking diversity (separation of 3 types of blocks) ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
import scipy.stats as stats
import cartopy

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

#%%
### read basic data ###
path_LWA_AC = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z_AC/*.nc")
path_LWA_AC.sort()
N=len(path_LWA_AC)   

path_LWA = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z/*.nc")
path_LWA.sort()
N=len(path_LWA)  

path_LWA_C = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA_Z_C/*.nc")
path_LWA_C.sort()
N=len(path_LWA_C)  

path_Z = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/Z500/*.nc")
path_Z.sort()
N=len(path_Z)

path_dT = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/tdt_moist/*.nc4")
path_dT.sort()
N=len(path_dT)

path_T = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/T/*.nc4")
path_T.sort()
N=len(path_T)

path_dA = glob.glob(r"/depot/wanglei/data/Reanalysis/MERRA2/LWA/*.nc")
path_dA.sort()
N=len(path_dA)

path_dA2 = glob.glob(r"/scratch/bell/liu3315/MERRA2/LWA_new/*.nc")
path_dA2.sort()
N=len(path_dA2)


### Read basic variables ###
file0 = Dataset(path_dT[0],'r')
lon = file0.variables['lon'][:]
lon[288]=0
lat = file0.variables['lat'][:]
plev = file0.variables['lev'][:]
lat_SH = lat[0:180]
lat_NH = lat[180:]
nplev = len(plev)
nlon = len(lon)
nlat = len(lat)
nlat_SH = len(lat_SH)
nlat_NH =len(lat_NH)
dlat = (lat[1]-lat[0])
dlon = (lon[1] - lon[0])
file0.close()

### read blocking data ###
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_date", "rb") as fp:
    Blocking_date = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_lat", "rb") as fp:
    Blocking_lat = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_lon", "rb") as fp:
    Blocking_lon = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_lon_wide", "rb") as fp:
    Blocking_lon_wide = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_area", "rb") as fp:
    Blocking_area = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_label", "rb") as fp:
    Blocking_label = pickle.load(fp)  
    
B_freq = np.load("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/B_freq.npy")

### Time Management ###
Datestamp = pd.date_range(start="1980-01-01",end="2023-05-01")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date0['date'].dt.month 
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Date = list(Date0['date'])
nday = len(Date)

#%%
### get the blocking peaking date and location and wave activity ###
Blocking_peaking_date = []
Blocking_peaking_date_index = []
Blocking_peaking_lon = []
Blocking_peaking_lat = []
Blocking_peaking_LWA = []
Blocking_duration =[]
Blocking_velocity = [] 
Blocking_peaking_lon_wide = []
Blocking_peaking_area = []
for n in np.arange(len(Blocking_date)):
    start = Date.index(Blocking_date[n][0])
    end = Date.index(Blocking_date[n][-1])
    duration = len(Blocking_date[n])
    LWA_event_max = np.zeros((duration))

    for d in np.arange(duration):
        index = start+d
        lo = np.squeeze(np.array(np.where( lon[:]==Blocking_lon[n][d])))
        la = np.squeeze(np.array(np.where( lat[:]==Blocking_lat[n][d])))    
        file_LWA = Dataset(path_LWA[index],'r')
        LWA_event_max[d]  = file_LWA.variables['LWA_Z500'][0,0,la,lo]
        file_LWA.close()
        
    Blocking_peaking_date_index=int(np.squeeze(np.array(np.where( LWA_event_max==np.max(LWA_event_max) ))))
    Blocking_peaking_LWA.append(np.max(LWA_event_max))
    Blocking_peaking_date.append(Blocking_date[n][Blocking_peaking_date_index])
    Blocking_peaking_lon.append(Blocking_lon[n][Blocking_peaking_date_index])
    Blocking_peaking_lat.append(Blocking_lat[n][Blocking_peaking_date_index])
    Blocking_peaking_lon_wide.append(Blocking_lon_wide[n][Blocking_peaking_date_index])
    Blocking_peaking_area.append(Blocking_area[n][Blocking_peaking_date_index])
    Blocking_duration.append(duration)
    Blocking_velocity.append( haversine(Blocking_lon[n][0], Blocking_lat[n][0], Blocking_lon[n][-1], Blocking_lat[n][-1])/(duration*24*60*60) )
    
    print(n)

### you may directly read  the data ###
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_date", "rb") as fp:
    Blocking_peaking_date = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_lat", "rb") as fp:
    Blocking_peaking_lat = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_lon", "rb") as fp:
    Blocking_peaking_lon = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_lon_wide", "rb") as fp:
    Blocking_peaking_lon_wide = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_area", "rb") as fp:
    Blocking_peaking_area = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_peaking_LWA", "rb") as fp:
    Blocking_peaking_LWA = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_velocity", "rb") as fp:
    Blocking_velocity = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_duration", "rb") as fp:
    Blocking_duration = pickle.load(fp)
    


#%%
###### Code for separate 3 types of blocks (ridge, trough, dipole) ######
#### Method: Focus on the peaking date, calculate the total LWA_AC and LWA_C of the block region ####
Blocking_ridge_date = [];  Blocking_ridge_lon = []; Blocking_ridge_lat=[];  Blocking_ridge_peaking_date = [];   Blocking_ridge_peaking_lon = []; Blocking_ridge_peaking_lat=[];  Blocking_ridge_duration = [];  Blocking_ridge_velocity = [];    Blocking_ridge_area = [];   Blocking_ridge_peaking_LWA = [];   Blocking_ridge_A = []; Blocking_ridge_C = [];   Blocking_ridge_label =[]
Blocking_trough_date = []; Blocking_trough_lon =[]; Blocking_trough_lat=[]; Blocking_trough_peaking_date = [];  Blocking_trough_peaking_lon =[]; Blocking_trough_peaking_lat=[]; Blocking_trough_duration = []; Blocking_trough_velocity = [];   Blocking_trough_area = [];  Blocking_trough_peaking_LWA = [];  Blocking_trough_A = []; Blocking_trough_C = []; Blocking_trough_label =[]
Blocking_dipole_date = []; Blocking_dipole_lon =[]; Blocking_dipole_lat=[]; Blocking_dipole_peaking_date = [];  Blocking_dipole_peaking_lon =[]; Blocking_dipole_peaking_lat=[]; Blocking_dipole_duration= []; Blocking_dipole_velocity =[];    Blocking_dipole_area = [];   Blocking_dipole_peaking_LWA = [];  Blocking_dipole_A = []; Blocking_dipole_C = []; Blocking_dipole_label = []
lat_range=int(int((90-np.max(Blocking_peaking_lat))/dlat)*2+1)
lon_range=int(30/dlon)+1


for n in np.arange(len(Blocking_lon)):

    LWA_AC_sum = 0
    LWA_C_sum = 0
    Blocking_A = []
    Blocking_C = []
        
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_peaking_date[n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_peaking_lon[n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_peaking_lat[n]))) 
    
    t = np.squeeze(np.where(np.array(Blocking_date[n]) == np.array(Blocking_peaking_date[n] )))
    
    file_LWA = Dataset(path_LWA[peaking_date_index],'r')
    LWA_max  = file_LWA.variables['LWA_Z500'][0,0,peaking_lat_index,peaking_lon_index]
    file_LWA.close() 
    

    ### date LWA_AC ###
    file_LWA_AC = Dataset(path_LWA_AC[peaking_date_index],'r')
    LWA_AC  = file_LWA_AC.variables['LWA_Z500'][0,0,180:,:]
    file_LWA_AC.close()
    
    ### date LWA_C ###
    file_LWA_C = Dataset(path_LWA_C[peaking_date_index],'r')
    LWA_C  = file_LWA_C.variables['LWA_Z500'][0,0,180:,:]
    file_LWA_C.close()
    
    LWA_AC = np.roll(LWA_AC, int(nlon/2)-peaking_lon_index, axis=1)
    LWA_C = np.roll(LWA_C,   int(nlon/2)-peaking_lon_index, axis=1)
    WE = np.roll(Blocking_label[n][t], int(nlon/2)-peaking_lon_index, axis=1)
    lon_roll = np.roll(lon,   int(nlon/2)-peaking_lon_index)

    LWA_AC = LWA_AC[  :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    LWA_C = LWA_C[    :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    WE = WE[   :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    
    LWA_AC_d = np.zeros((nlat_NH, lon_range))
    LWA_C_d = np.zeros((nlat_NH, lon_range))
    LWA_AC_d[WE == True]  = LWA_AC[WE== True]
    LWA_C_d[WE == True]  =  LWA_C[WE == True]
    
    
    LWA_AC_sum += LWA_AC_d.sum()
    LWA_C_sum += LWA_C_d.sum()
    Blocking_A.append(LWA_AC_d.sum())
    Blocking_C.append(LWA_C_d.sum())

### if the anticyclonic LWA is much stronger than cytclonic LWA, then it is defined as ridge ###
### if the anticyclonic LWA is comparable with cyclonic LWA, then it is defined as dipole ###
### if the anticyclonic LWA is weaker than cyclonic LWA, then it is defined as trough events ###
    if LWA_AC_sum > 10 * LWA_C_sum :
        Blocking_ridge_date.append(Blocking_date[n]);                 Blocking_ridge_lon.append(Blocking_lon[n]);                  Blocking_ridge_lat.append(Blocking_lat[n])
        Blocking_ridge_peaking_date.append(Blocking_peaking_date[n]); Blocking_ridge_peaking_lon.append(Blocking_peaking_lon[n]);  Blocking_ridge_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_ridge_peaking_LWA.append(LWA_max)
        Blocking_ridge_duration.append(len(Blocking_date[n]));        Blocking_ridge_velocity.append(Blocking_velocity[n]);        Blocking_ridge_area.append(Blocking_peaking_area[n]); Blocking_ridge_label.append(Blocking_label[n])
        Blocking_ridge_A.append(Blocking_A);                          Blocking_ridge_C.append(Blocking_C)
    elif LWA_C_sum > 2 * LWA_AC_sum:
        Blocking_trough_date.append(Blocking_date[n]);                 Blocking_trough_lon.append(Blocking_lon[n]);                 Blocking_trough_lat.append(Blocking_lat[n])
        Blocking_trough_peaking_date.append(Blocking_peaking_date[n]); Blocking_trough_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_trough_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_trough_peaking_LWA.append(LWA_max)
        Blocking_trough_duration.append(len(Blocking_date[n]));        Blocking_trough_velocity.append(Blocking_velocity[n]);       Blocking_trough_area.append(Blocking_peaking_area[n]); Blocking_trough_label.append(Blocking_label[n])
        Blocking_trough_A.append(Blocking_A);                          Blocking_trough_C.append(Blocking_C)
    else:
        Blocking_dipole_date.append(Blocking_date[n]);                 Blocking_dipole_lon.append(Blocking_lon[n]);                 Blocking_dipole_lat.append(Blocking_lat[n])
        Blocking_dipole_peaking_date.append(Blocking_peaking_date[n]); Blocking_dipole_peaking_lon.append(Blocking_peaking_lon[n]); Blocking_dipole_peaking_lat.append(Blocking_peaking_lat[n]); Blocking_dipole_peaking_LWA.append(LWA_max)
        Blocking_dipole_duration.append(len(Blocking_date[n]));        Blocking_dipole_velocity.append(Blocking_velocity[n]);       Blocking_dipole_area.append(Blocking_peaking_area[n]); Blocking_dipole_label.append(Blocking_label[n])
        Blocking_dipole_A.append(Blocking_A);                          Blocking_dipole_C.append(Blocking_C)

    print(n)
    
Blocking_diversity_date= [];   Blocking_diversity_lon= []; Blocking_diversity_lat= []; Blocking_diversity_date= []; Blocking_diversity_peaking_date= []; Blocking_diversity_peaking_lon= [];  Blocking_diversity_peaking_lat=[]; Blocking_diversity_peaking_LWA=[]; Blocking_diversity_duration=[]; Blocking_diversity_area=[]; Blocking_diversity_velocity=[]; Blocking_diversity_A = []; Blocking_diversity_C = []; Blocking_diversity_label = []
Blocking_diversity_date.append(Blocking_ridge_date);   Blocking_diversity_lon.append(Blocking_ridge_lon); Blocking_diversity_lat.append(Blocking_ridge_lat); Blocking_diversity_peaking_date.append(Blocking_ridge_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_ridge_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_ridge_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_ridge_peaking_LWA); Blocking_diversity_velocity.append(Blocking_ridge_velocity); Blocking_diversity_duration.append(Blocking_ridge_duration); Blocking_diversity_area.append(Blocking_ridge_area);  Blocking_diversity_A.append(Blocking_ridge_A) ;         Blocking_diversity_C.append(Blocking_ridge_C)   ;  Blocking_diversity_label.append(Blocking_ridge_label)
Blocking_diversity_date.append(Blocking_trough_date);  Blocking_diversity_lon.append(Blocking_trough_lon); Blocking_diversity_lat.append(Blocking_trough_lat); Blocking_diversity_peaking_date.append(Blocking_trough_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_trough_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_trough_peaking_lon);Blocking_diversity_peaking_LWA.append(Blocking_trough_peaking_LWA); Blocking_diversity_velocity.append(Blocking_trough_velocity); Blocking_diversity_duration.append(Blocking_trough_duration); Blocking_diversity_area.append(Blocking_trough_area);  Blocking_diversity_A.append(Blocking_trough_A) ; Blocking_diversity_C.append(Blocking_trough_C) ;  Blocking_diversity_label.append(Blocking_trough_label)    
Blocking_diversity_date.append(Blocking_dipole_date);  Blocking_diversity_lon.append(Blocking_dipole_lon); Blocking_diversity_lat.append(Blocking_dipole_lat); Blocking_diversity_peaking_date.append(Blocking_dipole_peaking_date); Blocking_diversity_peaking_lat.append(Blocking_dipole_peaking_lat); Blocking_diversity_peaking_lon.append(Blocking_dipole_peaking_lon); Blocking_diversity_peaking_LWA.append(Blocking_dipole_peaking_LWA); Blocking_diversity_velocity.append(Blocking_dipole_velocity); Blocking_diversity_duration.append(Blocking_dipole_duration); Blocking_diversity_area.append(Blocking_dipole_area); Blocking_diversity_A.append(Blocking_dipole_A) ; Blocking_diversity_C.append(Blocking_dipole_C) ;  Blocking_diversity_label.append(Blocking_dipole_label)


### you can directly read the data ###
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_date2", "rb") as fp:
    Blocking_diversity_date = pickle.load(fp)      
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_lon2", "rb") as fp:
    Blocking_diversity_lon = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_lat2", "rb") as fp:
    Blocking_diversity_lat = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_peaking_date2", "rb") as fp:
    Blocking_diversity_peaking_date = pickle.load(fp)     
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_peaking_lon2", "rb") as fp:
    Blocking_diversity_peaking_lon = pickle.load(fp)      
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_peaking_lat2", "rb") as fp:
    Blocking_diversity_peaking_lat = pickle.load(fp)  
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_peaking_LWA2", "rb") as fp:
    Blocking_diversity_peaking_LWA = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_duration2", "rb") as fp:
    Blocking_diversity_duration = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_velocity2", "rb") as fp:
    Blocking_diversity_velocity = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_area2", "rb") as fp:
    Blocking_diversity_area = pickle.load(fp)     
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Blocking_diversity_label2", "rb") as fp:
    Blocking_diversity_label = pickle.load(fp)   


#################################################################################################################
#%%
###### Code for doing the horizontal composite ######
LWA_A_Blocking_diversity_com = []
LWA_C_Blocking_diversity_com = []
dAdt_A_Blocking_diversity_com = []
dAdt_C_Blocking_diversity_com = []
dTdt_Blocking_diversity_com = []
for i in np.array([0,1, 2]):
    lat_range=int(int((90-np.max(Blocking_diversity_peaking_lat[i]))/dlat)*2+1); lon_range=int(60/dlon)+1
    LWA_A_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    LWA_C_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_A_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dAdt_C_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))
    dTdt_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),lat_range, lon_range))

    for n in np.arange(len(Blocking_diversity_peaking_date[i])):
        
        LWA_A_d = np.zeros((nlat,nlon)); LWA_C_d = np.zeros((nlat,nlon)); dTdt_d = np.zeros((nlat,nlon))
        dAdt_A_d = np.zeros((nlat,nlon)); dAdt_C_d = np.zeros((nlat,nlon))
        
        ### peaking date information ###
        peaking_date_index = Date.index(Blocking_diversity_peaking_date[i][n])
        peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
        peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))
        
        file_dAdt = Dataset(path_dA2[peaking_date_index],'r')

        LWA_A_d[:,:] = file_dAdt.variables['LWA_A_column'][0,:,:]
        LWA_C_d[:,:] = file_dAdt.variables['LWA_C_column'][0,:,:]
        dAdt_A_d[:,:] = file_dAdt.variables['dAdt_moist_A_column'][0,:,:]
        dAdt_C_d[:,:] = file_dAdt.variables['dAdt_moist_C_column'][0,:,:]
        dTdt_d = file_dAdt.variables['dTdt_moist_column'][0,:,:] #5 875hPa, 6 850hPa, 14 600hPa, 16 500hPa


        ### shift the array to make the conpoiste
        LWA_A_d = np.roll(LWA_A_d, int(nlon/2)-peaking_lon_index, axis=1)
        LWA_C_d = np.roll(LWA_C_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_A_d = np.roll(dAdt_A_d, int(nlon/2)-peaking_lon_index, axis=1)
        dAdt_C_d = np.roll(dAdt_C_d, int(nlon/2)-peaking_lon_index, axis=1)
        dTdt_d = np.roll(dTdt_d, int(nlon/2)-peaking_lon_index, axis=1)

        
        LWA_A_Blocking_diversity[n,:,:] = LWA_A_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,    int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        LWA_C_Blocking_diversity[n,:,:] = LWA_C_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,    int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dTdt_Blocking_diversity[n,:,:] = dTdt_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_A_Blocking_diversity[n,:,:] = dAdt_A_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        dAdt_C_Blocking_diversity[n,:,:] = dAdt_C_d[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]

        print(n)
        
    LWA_A_Blocking_diversity_m = LWA_A_Blocking_diversity.mean(axis=0)
    LWA_C_Blocking_diversity_m = LWA_C_Blocking_diversity.mean(axis=0)
    dAdt_A_Blocking_diversity_m = np.nanmean(dAdt_A_Blocking_diversity, axis=0)
    dAdt_C_Blocking_diversity_m = np.nanmean(dAdt_C_Blocking_diversity, axis=0)
    dTdt_Blocking_diversity_m = np.nanmean(dTdt_Blocking_diversity, axis=0)

    
    LWA_A_Blocking_diversity_com.append(LWA_A_Blocking_diversity_m)
    LWA_C_Blocking_diversity_com.append(LWA_C_Blocking_diversity_m)
    dAdt_A_Blocking_diversity_com.append(dAdt_A_Blocking_diversity_m)
    dAdt_C_Blocking_diversity_com.append(dAdt_C_Blocking_diversity_m)
    dTdt_Blocking_diversity_com.append(dTdt_Blocking_diversity_m)


with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_A_Blocking_diversity_com2", "wb") as fp:
    pickle.dump(LWA_A_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_C_Blocking_diversity_com2", "wb") as fp:
    pickle.dump(LWA_C_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_A_Blocking_diversity_com2", "wb") as fp:
    pickle.dump(dAdt_A_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_C_Blocking_diversity_com2", "wb") as fp:
    pickle.dump(dAdt_C_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_Blocking_diversity_com2", "wb") as fp:
    pickle.dump(dTdt_Blocking_diversity_com, fp) 

#################################################################################################################
#%%
###### Code for doing the cross section composite ######
nzlev=47
nplev=42

lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1

dTdt_diversity_cros_com=[]
dAdt_diversity_cros_com=[]
LWA_diversity_cros_com=[]

for i in np.array([0,1, 2]):
    dAdt_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    LWA_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon))
    dTdt_diversity_cros = np.zeros((len(Blocking_diversity_date[i]),nzlev,nlon)) 

    for n in np.arange(len(Blocking_diversity_peaking_date[i])):
        ### peaking date information ###
        peaking_date_index = Date.index(Blocking_diversity_peaking_date[i][n])+3
        peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
        peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))
        
        file_dA = Dataset(path_dA2[peaking_date_index],'r')
        LWA1 = file_dA.variables['LWA'][0,:,:,:] 
        dAdt1 = file_dA.variables['dAdt_moist'][0,:,:,:] 
        dTdt1 = file_dA.variables['dTdt_moist'][0,:,:,:] 
        file_dA.close()

        ### shift the array to make the conpoiste ###
        dAdt2 = np.roll(dAdt1, int(nlon/2)-peaking_lon_index, axis=2)
        dTdt2 = np.roll(dTdt1, int(nlon/2)-peaking_lon_index, axis=2)
        LWA2 = np.roll(LWA1, int(nlon/2)-peaking_lon_index, axis=2)
        lon1 = np.roll(lon, int(nlon/2)-peaking_lon_index)
        
        ### get a +- 30 lon wide domain, average the +-10 latitudes, now it's a Z_lon cross section ###
        dAdt2 = np.mean(dAdt2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        dTdt2 = np.mean(dTdt2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        LWA2 = np.mean(LWA2[:,peaking_lat_index-lat_range:peaking_lat_index+lat_range+1,  :],axis=1)
        lat2 =  lat[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1]
        lon2 = lon1[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
        
        dAdt_diversity_cros[n,:,:] = dAdt2
        dTdt_diversity_cros[n,:,:] = dTdt2
        LWA_diversity_cros[n,:,:] = LWA2
        print(n)

    dAdt_diversity_cros_m = dAdt_diversity_cros.mean(axis=0)    
    dTdt_diversity_cros_m = np.nanmean(dTdt_diversity_cros,axis=0)
    LWA_diversity_cros_m = LWA_diversity_cros.mean(axis=0)
    
    dTdt_diversity_cros_com.append(dTdt_diversity_cros_m)
    dAdt_diversity_cros_com.append(dAdt_diversity_cros_m)
    LWA_diversity_cros_com.append(LWA_diversity_cros_m)


with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_diversity_cros_com2_lag3", "wb") as fp:
    pickle.dump(dAdt_diversity_cros_com, fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_diversity_cros_com2_lag3", "wb") as fp:
    pickle.dump(dTdt_diversity_cros_com, fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_diversity_cros_com2_lag3", "wb") as fp:
    pickle.dump(LWA_diversity_cros_com, fp)
    
    
#################################################################################################################
#%%
###### Code for Hovmoller calculation of three types of blocks ######
lat_range=int(7.5/dlat)+1
lon_range=int(60/dlon)+1
duration = 6  ### We will plot the +-6 days after and before the blocking peaking date ###
LWA_Blocking_diversity_com=[]
dAdt_Blocking_diversity_com=[]
dTdt_Blocking_diversity_com=[]

for i in np.array([0,1,2]):
    
    LWA_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),13,nlon))
    dTdt_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),13,nlon)) 
    dAdt_Blocking_diversity = np.zeros((len(Blocking_diversity_date[i]),13,nlon)) 
    ti=-1  
    for n in np.arange(1,len(Blocking_diversity_date[i])-1 ):
        ti+=1
        LWA_d = np.zeros((2*duration+1,nlat,nlon))
        dTdt_d = np.zeros((2*duration+1,nlat,nlon))
        dAdt_d = np.zeros((2*duration+1,nlat,nlon))
        dTdt_0 = np.zeros((2*duration+1, nplev,nlat,nlon))

        for j in np.arange(2*duration+1):                
            index = Date.index(Blocking_diversity_peaking_date[i][n])-duration+j
            file_LWA = Dataset(path_LWA[index],'r')
            file_dA = Dataset(path_dA[index],'r')
            file_dTdt = Dataset(path_dT[index],'r')
            LWA_d[j,:,:] = file_LWA.variables['LWA_Z500'][0,:,:]
            dAdt_d[j,:,:] = file_dA.variables['dAdt_moist_column'][0,:,:]
            dTdt_0 = file_dTdt.variables['DTDTMST'][0,:,:,:]
            file_LWA.close()
            file_dTdt.close()
            file_dA.close()
        
            ### Don't forget to mask the dTdt data ###
            dTdt_0[dTdt_0>1] = np.nan   
            
            ### Make the column average of dTdt ###
            rho = np.array([plev[i]/1000 for i in np.arange(nplev)]) 
            rho = rho[:,np.newaxis,np.newaxis] * np.ones((nplev,nlat,nlon))  ## Density in 3D ##
            dTdt_0 = dTdt_0[1:nplev-1,:,:] * rho[1:nplev-1]                        ## we only use 1km-47km interia points
            dTdt_d[j,:,:] = np.nansum(dTdt_0[:,:,:],axis=0)/ rho[1:nplev-1].sum(axis=0) 
        
        lon_c = Blocking_diversity_peaking_lon[i][n]; lat_c = Blocking_diversity_peaking_lat[i][n]
        lon_c_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
        lat_c_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))

        ## Based on this latitude, we do a meridional average +- 5 latitudes to represent the value for each longitude ##
        LWA_lon = LWA_d[:,lat_c_index-lat_range:lat_c_index+lat_range+1,:].mean(axis=1)
        dTdt_lon = dTdt_d[:,lat_c_index-lat_range:lat_c_index+lat_range+1,:].mean(axis=1)
        dAdt_lon = dAdt_d[:,lat_c_index-lat_range:lat_c_index+lat_range+1,:].mean(axis=1)

        ## For visulization, we will put the center lontitude to the center, so we need to reorganize the array ##
        LWA_lon_plot = np.roll(LWA_lon, int(nlon/2)-lon_c_index, axis=1)
        dTdt_lon_plot = np.roll(dTdt_lon, int(nlon/2)-lon_c_index, axis=1)
        dAdt_lon_plot = np.roll(dAdt_lon, int(nlon/2)-lon_c_index, axis=1)

        LWA_Blocking_diversity[ti,:,:] = LWA_lon_plot
        dTdt_Blocking_diversity[ti,:,:] = dTdt_lon_plot
        dAdt_Blocking_diversity[ti,:,:] = dAdt_lon_plot
        
        print(n)
    
    LWA_Blocking_diversity_m = LWA_Blocking_diversity.mean(axis=0)
    dAdt_Blocking_diversity_m = dAdt_Blocking_diversity.mean(axis=0)
    dTdt_Blocking_diversity_m = dTdt_Blocking_diversity.mean(axis=0)
    
    LWA_Blocking_diversity_com.append(LWA_Blocking_diversity_m) 
    dAdt_Blocking_diversity_com.append(dAdt_Blocking_diversity_m)
    dTdt_Blocking_diversity_com.append(dTdt_Blocking_diversity_m)

    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_Hov2", "wb") as fp:
    pickle.dump(LWA_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_Hov2", "wb") as fp:
    pickle.dump(dAdt_Blocking_diversity_com, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_Hov2", "wb") as fp:
    pickle.dump(dTdt_Blocking_diversity_com, fp) 
    
#################################################################################################################    
#%%    
### Code for getting the domain averaged dAdt_moist and dTdt_moist of each blocking event  ###
i = 1  
dTdt_max_list = []
dAdt_max_list = []
lat_range=int(15/dlat)+1; lon_range=int(15/dlon)+1
for n in np.arange(len(Blocking_diversity_peaking_date[i])):
        
    ### peaking date information ###
    peaking_date_index = Date.index(Blocking_diversity_peaking_date[i][n])
    peaking_lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_peaking_lon[i][n])))
    peaking_lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_peaking_lat[i][n])))
    
    file_dTdt = Dataset(path_dT[peaking_date_index],'r')
    file_Z = Dataset(path_Z[peaking_date_index],'r')
    Z1 = file_Z.variables['H'][0,0,:,:]
    file_dAdt = Dataset(path_dA[peaking_date_index],'r')
    dTdt1 = file_dTdt.variables['DTDTMST'][0,:,:,:] #5 875hPa, 6 850hPa, 14 600hPa, 16 500hPa
    dAdt1 = file_dAdt.variables['dAdt_moist_column'][0,:,:] #5 875hPa, 6 850hPa, 14 600hPa, 16 500hPa
    file_Z.close()
    file_dTdt.close()
    
    ### Don't forget to mask the dTdt data ###
    dTdt1[dTdt1>1] = np.nan
    rho = np.array([plev[i]/1000 for i in np.arange(nplev)]) 
    rho = rho[:,np.newaxis,np.newaxis] * np.ones((nplev,nlat,nlon))  ## Density in 3D ##
    dTdt1 = dTdt1[1:nplev-1,:,:] * rho[1:nplev-1]                        ## we only use 1km-47km interia points
    dTdt1 = np.nansum(dTdt1[:,:,:],axis=0)/ rho[1:nplev-1].sum(axis=0) 
    # dTdt11[dTdt11>1] = np.nan
    
    ### shift the array to make the conpoiste
    dTdt1 = np.roll(dTdt1, int(nlon/2)-peaking_lon_index, axis=1)
    dAdt1 = np.roll(dAdt1, int(nlon/2)-peaking_lon_index, axis=1)
    Z1 = np.roll(Z1, int(nlon/2)-peaking_lon_index, axis=1)
    lon1 = np.roll(lon, int(nlon/2)-peaking_lon_index)
    
    ### get a +- 30 lon wide domain ###
    dTdt2 = dTdt1[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    dAdt2 = dAdt1[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    Z2 =    Z1[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1,  int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    lat2 =  lat[peaking_lat_index-int(lat_range/2):peaking_lat_index+int(lat_range/2)+1]; lon2 = lon1[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
    
    
    dTdt_max_list.append(np.mean(dTdt2))
    dAdt_max_list.append(np.mean(dAdt2))
    
    print(n)

#%%
dAdt_ridge_list = dAdt_max_list
dTdt_ridge_list = dTdt_max_list

dAdt_dipole_list = dAdt_max_list
dTdt_dipole_list = dTdt_max_list

dAdt_trough_list = dAdt_max_list
dTdt_trough_list = dTdt_max_list

with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_ridge_list2", "wb") as fp:
    pickle.dump(dAdt_ridge_list, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_ridge_list2", "wb") as fp:
    pickle.dump(dTdt_ridge_list, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_dipole_list2", "wb") as fp:
    pickle.dump(dAdt_dipole_list, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_dipole_list2", "wb") as fp:
    pickle.dump(dTdt_dipole_list, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_trough_list2", "wb") as fp:
    pickle.dump(dAdt_trough_list, fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_trough_list2", "wb") as fp:
    pickle.dump(dTdt_trough_list, fp)   



#################################################################################################################
#%%
### Figure 1 ###
fig = plt.figure(figsize=[10,6])
proj=ccrs.PlateCarree(central_longitude=180)
gx = fig.add_subplot(1,1,1, projection=proj)
gx.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], transform=ccrs.PlateCarree() ,s= 5, label="ridge blocks", color='r', alpha=1)  
gx.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], transform=ccrs.PlateCarree() ,s= 5, label="dipole blocks",color='b',alpha=1)  
plt.xlabel('longitude',fontsize=12)
plt.ylabel('latitude',fontsize=12)     
plt.title("Geographic Distribution of Ridge Blocks and Dipole Blokcs", pad=5, fontdict={'family':'Times New Roman', 'size':12})
gx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.6)
gx.coastlines()
gx.gridlines(linestyle="--", alpha=0.7)
gx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
plt.legend(loc='lower left')
gx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
gx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
gx.xaxis.set_major_formatter(lon_formatter)
gx.yaxis.set_major_formatter(lat_formatter) 
# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/Figure1.png",dpi=600)

#################################################################################################################
#%%
###### Figure 2 ######
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_Blocking_diversity_com2", "rb") as fp:
    LWA_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_Blocking_diversity_com2", "rb") as fp:
    dTdt_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Z_Blocking_diversity_com2", "rb") as fp:
    Z_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_diversity_cros_com2", "rb") as fp:
    dTdt_diversity_cros_com = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_diversity_cros_com2", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_Hov2", "rb") as fp:
    LWA_Hov = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_Hov2", "rb") as fp:
    dTdt_Hov = pickle.load(fp)
           
minlev = Z_Blocking_diversity_com[2].min()
maxlev = Z_Blocking_diversity_com[2].max()
levs_Z = np.linspace(5400,5870,11)

minlev = dTdt_Blocking_diversity_com[2].min()
maxlev = dTdt_Blocking_diversity_com[2].max()
levs_dTdt = np.linspace(0, 2e-5,15)
lon_range=int(60/dlon)+1 

minlev = dTdt_Hov[0].min()
maxlev = dTdt_Hov[0].max()
levs_dTdt_hov = np.linspace(0, 2e-5,15)
 
duration = 6
t_range=6

maxlevel = LWA_Blocking_diversity_com[0][:,:].max()
minlevel = LWA_Blocking_diversity_com[0][:,:].min()  
levs_LWA= np.linspace(0.7e8, 2e8, 11)

fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(3,2,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[0])), dTdt_Blocking_diversity_com[0], levs_dTdt, cmap='hot_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[0])), Z_Blocking_diversity_com[0], levs_Z, colors='k')  
ax.set_yticks([0,25,50, 75,100])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,16,32,48,64,80,96])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Composite Ridge Blocks \n(a)", pad=5, fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(3,2,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dTdt_Blocking_diversity_com[2])), dTdt_Blocking_diversity_com[2], levs_dTdt, cmap='hot_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_com[2])), Z_Blocking_diversity_com[2], levs_Z, colors='k')  
bx.set_yticks([0,19.5,39, 58.5,78])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,16,32,48,64,80,96])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Composite Dipole Blocks \n(b)" , pad=5, fontsize=12)
cbar = fig.add_axes([0.93,0.64,0.01,0.25])
cb = plt.colorbar(a, cax=cbar, ticks=[0,0.5e-5,1e-5,1.5e-5,2e-5]) 
cb.set_ticklabels(['0','0.5','1','1.5','2'])
cb.set_label('moist-induced diabatic heating ($10^{-5}$K/s)',fontsize=10)

maxlevel = np.max(dTdt_diversity_cros_com[2])
minlevel = np.min(dTdt_diversity_cros_com[2]) 
levs_dTdt_cross = np.linspace(0, 2.5e-5, 15)

maxlevel = np.max(LWA_diversity_cros_com[2])
minlevel = np.min(LWA_diversity_cros_com[2]) 
levs_LWA_cross = np.linspace(0, 200, 11)

zlev = np.arange(1000,31000,1000)
zlev1=0; zlev2=12

cx = fig.add_subplot(3,2,3)
c=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
plt.contour(np.arange(0,lon_range), zlev[zlev1:zlev2],   LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
plt.ylabel('height (m)',fontsize=12)
cx.set_yticks([1000,3000,5000,7000,9000,11000])
cx.set_xticks([0,16,32,48,64,80,96])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(c)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

dx = fig.add_subplot(3,2,4)
c=plt.contourf(np.arange(0,lon_range), zlev[zlev1:zlev2], dTdt_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_cross,cmap='hot_r',extend ='both')
plt.contour(np.arange(0,lon_range), zlev[zlev1:zlev2],   LWA_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k")
dx.set_yticks([1000,3000,5000,7000,9000,11000])
dx.set_xticks([0,16,32,48,64,80,96])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(d)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.37,0.01,0.25])
cb = plt.colorbar(c, cax=cbar, ticks=[0,0.5e-5,1e-5,1.5e-5,2e-5,2.5e-5]) 
cb.set_ticklabels(['0','0.5','1','1.5','2', '2.5'])
cb.set_label('moist-induced diabatic heating  ($10^{-5}$K/s)',fontsize=10)


lon_range=int(180/dlon)+1
ex = fig.add_subplot(3,2,5)
ex.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA, colors="k", linewidths=0.7)
ex.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dTdt_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_hov, cmap='hot_r',extend ='both')
ex.set_ylabel('lag (days)',fontsize=10)
ex.set_yticks([0,3,6,9,12])
ex.set_yticklabels([-6,-3,0,3,6])
ex.set_xticks([-90,-60,-30,0,30,60,90])
ex.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(e)", pad=5, fontdict={'family':'Times New Roman', 'size':12})


fx = fig.add_subplot(3,2,6)
fx.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA, colors="k", linewidths=0.7)
h2 = fx.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dTdt_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dTdt_hov, cmap='hot_r',extend ='both')
fx.set_ylabel('time (days)',fontsize=10)
fx.set_yticks([0,3,6,9,12])
fx.set_yticklabels([-6,-3,0,3,6])
fx.set_xticks([-90,-60,-30,0,30,60,90])
fx.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(f)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

cbar = fig.add_axes([0.93,0.1,0.01,0.25])
cb = plt.colorbar(h2, cax=cbar, ticks=[0,0.5e-5,1e-5,1.5e-5, 2e-5]) 
cb.set_ticklabels(['0','0.5','1','1.5','2'])
cb.set_label('moist-induced diabatic heating  ($10^{-5}$K/s)',fontsize=10)

# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/Figure2.png",dpi=600)

#################################################################################################################
#%%
###### Figure 4 ######
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_Blocking_diversity_com2", "rb") as fp:
    dAdt_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Z_Blocking_diversity_com2", "rb") as fp:
    Z_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_diversity_cros_com2", "rb") as fp:
    dAdt_diversity_cros_com = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_diversity_cros_com2", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_Hov2", "rb") as fp:
    LWA_Hov = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_Hov2", "rb") as fp:
    dAdt_Hov = pickle.load(fp)
        
minlev = Z_Blocking_diversity_com[0].min()
maxlev = Z_Blocking_diversity_com[0].max()
levs_Z = np.linspace(5400,5870,11)

minlev = dAdt_Blocking_diversity_com[0].min()
maxlev = dAdt_Blocking_diversity_com[0].max()
levs_dAdt = np.linspace(-4.5e-5, 4.5e-5,21)
lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1
duration = 6  
  
fig = plt.figure(figsize=[10,12])
ax = fig.add_subplot(3,2,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[0])), dAdt_Blocking_diversity_com[0], levs_dAdt, cmap='RdBu_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[0])), Z_Blocking_diversity_com[0], levs_Z, colors='k', alpha = 0.7)  
ax.set_yticks([0,25,50, 75,100])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,16,32,48,64,80,96])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Composite Ridge Blocks \n(a)", pad=5, fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(3,2,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[2])), dAdt_Blocking_diversity_com[2], levs_dAdt, cmap='RdBu_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_com[2])), Z_Blocking_diversity_com[2], levs_Z, colors='k', alpha = 0.7)  
bx.set_yticks([0,20,40, 60,80])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,16,32,48,64,80,96])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Composite Dipole Blocks \n(b)" , pad=5, fontsize=12) 
cbar = fig.add_axes([0.93,0.64,0.01,0.25])
cb = plt.colorbar(a, cax=cbar, ticks=[-4e-5,-3e-5,-2e-5,-1e-5,0,1e-5,2e-5,3e-5,4e-5]) 
cb.set_ticklabels(['-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)

maxlevel = np.max(dAdt_diversity_cros_com[0])
minlevel = np.min(dAdt_diversity_cros_com[0]) 
levs_dAdt_cross = np.linspace(-2e-4, 2e-4, 24)

maxlevel = np.max(LWA_diversity_cros_com[1])
minlevel = np.min(LWA_diversity_cros_com[1]) 
levs_LWA_cross = np.linspace(0, 200, 11)

zlev = np.arange(1000,31000,1000)
zlev1=0; zlev2=12

cx = fig.add_subplot(3,2,3)
c=plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", alpha = 0.7)
plt.ylabel('height (m)',fontsize=12)
cx.set_ylim(2000,10000)
cx.set_yticks([2000, 4000, 6000, 8000, 10000])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(c)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

dx = fig.add_subplot(3,2,4)
d = plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", alpha = 0.7)
dx.set_ylim(2000,10000)
dx.set_yticks([2000, 4000, 6000, 8000, 10000])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(d)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.37,0.01,0.25])
cb = plt.colorbar(c, cax=cbar, ticks=[-2e-4,-1.5e-4,-1e-4,-0.5e-4, 0, 0.5e-4, 1e-4, 1.5e-4, 2e-4]) 
cb.set_ticklabels(['-2','-1.5','-1','0.5','0','0.5','1','1.5','2'])
cb.set_label('moist-induced LWA tendency ($10^{-4}$m/$s^2$)',fontsize=10)


lat_range=int(10/dlat)+1
lon_range=int(180/dlon)+1
t_range=6
duration = 6
maxlevel = LWA_Hov[2][:,:].max()
minlevel = LWA_Hov[2][:,:].min()  
levs_LWA_Hov = np.linspace(0.7e8, 2e8, 11)
maxlevel = dAdt_Hov[2][:,:].max()
minlevel = dAdt_Hov[2][:,:].min()  
levs_dAdt_Hov = np.linspace(-5e-5, 5e-5, 11)
        
ex = fig.add_subplot(3,2,5)
ex.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.7)
ex.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
ex.set_ylabel('lag (days)',fontsize=12)
ex.set_yticks([0,3,6,9,12])
ex.set_yticklabels([-6,-3,0,3,6])
ex.set_xticks([-90,-60,-30,0,30,60,90])
ex.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(e)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

fx = fig.add_subplot(3,2,6)
fx.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.7)
f = fx.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
fx.set_yticks([0,3,6,9,12])
fx.set_yticklabels([-6,-3,0,3,6])
fx.set_xticks([-90,-60,-30,0,30,60,90])
fx.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(f)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.1,0.01,0.25])
cb = plt.colorbar(f, cax=cbar, ticks=[-5e-5,-4e-5, -3e-5,-2e-5,-1e-5, 0 ,1e-5,2e-5,3e-5,4e-5,5e-5]) 
cb.set_ticklabels(['-5.0','-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0','5.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)
# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/Figure4.png",dpi=600)

#################################################################################################################
#%%
###### Figure S5  ######
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_Blocking_diversity_com2", "rb") as fp:
    dAdt_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/Z_Blocking_diversity_com2", "rb") as fp:
    Z_Blocking_diversity_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_diversity_cros_com2", "rb") as fp:
    dAdt_diversity_cros_com = pickle.load(fp)    
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_diversity_cros_com2", "rb") as fp:
    LWA_diversity_cros_com = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/LWA_Hov2", "rb") as fp:
    LWA_Hov = pickle.load(fp)
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_Hov2", "rb") as fp:
    dAdt_Hov = pickle.load(fp)
        
minlev = Z_Blocking_diversity_com[0].min()
maxlev = Z_Blocking_diversity_com[0].max()
levs_Z = np.linspace(5400,5870,11)
levs_Z2 = np.linspace(5107,5700,11)

minlev = dAdt_Blocking_diversity_com[0].min()
maxlev = dAdt_Blocking_diversity_com[0].max()
levs_dAdt = np.linspace(-4.5e-5, 4.5e-5,21)
lat_range=int(10/dlat)+1
lon_range=int(60/dlon)+1
duration = 6  
  
fig = plt.figure(figsize=[12,12])
ax = fig.add_subplot(3,3,1)
a = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[0])), dAdt_Blocking_diversity_com[0], levs_dAdt, cmap='RdBu_r',extend='both')  
ax.contour(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[0])), Z_Blocking_diversity_com[0], levs_Z, colors='k', alpha = 0.7)  
ax.set_yticks([0,25,50, 75,100])
ax.set_yticklabels(['-20','-10','lat_c','+10','+20'])
ax.set_xticks([0,16,32,48,64,80,96])
ax.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
ax.set_title("Composite Ridge Blocks \n(a)", pad=5, fontsize=12)
ax.set_ylabel('relative latitude',fontsize=12)     

bx = fig.add_subplot(3,3,2)
b = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[2])), dAdt_Blocking_diversity_com[2], levs_dAdt, cmap='RdBu_r',extend='both')  
bx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_com[2])), Z_Blocking_diversity_com[2], levs_Z, colors='k', alpha = 0.7)  
bx.set_yticks([0,20,40, 60,80])
bx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bx.set_xticks([0,16,32,48,64,80,96])
bx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bx.set_title("Composite Dipole Blocks \n(b)" , pad=5, fontsize=12)
 

bbx = fig.add_subplot(3,3,3)
bb = plt.contourf(np.arange(0,lon_range),np.arange(0,len(dAdt_Blocking_diversity_com[1])), dAdt_Blocking_diversity_com[1], levs_dAdt, cmap='RdBu_r',extend='both')  
bbx.contour(np.arange(0,lon_range),np.arange(0,len(Z_Blocking_diversity_com[1])), Z_Blocking_diversity_com[1], levs_Z2, colors='k', alpha = 0.7)  
bbx.set_yticks([0,13.5,27, 40.5,54])
bbx.set_yticklabels(['-20','-10','lat_c','+10','+20'])
bbx.set_xticks([0,16,32,48,64,80,96])
bbx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
bbx.set_title("Composite Trough Events \n(c)" , pad=5, fontsize=12)
cbar = fig.add_axes([0.93,0.64,0.01,0.25])
cb = plt.colorbar(a, cax=cbar, ticks=[-4e-5,-3e-5,-2e-5,-1e-5,0,1e-5,2e-5,3e-5,4e-5]) 
cb.set_ticklabels(['-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)


maxlevel = np.max(dAdt_diversity_cros_com[0])
minlevel = np.min(dAdt_diversity_cros_com[0]) 
levs_dAdt_cross = np.linspace(-2e-4, 2e-4, 24)

maxlevel = np.max(LWA_diversity_cros_com[1])
minlevel = np.min(LWA_diversity_cros_com[1]) 
levs_LWA_cross = np.linspace(0, 200, 11)

zlev = np.arange(1000,31000,1000)
zlev1=0; zlev2=12

cx = fig.add_subplot(3,3,4)
c=plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[0][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", alpha = 0.7)
plt.ylabel('height (m)',fontsize=12)
cx.set_ylim(2000,10000)
cx.set_yticks([2000,4000,6000,8000,10000])
cx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(d)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

dx = fig.add_subplot(3,3,5)
d = plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[2][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", alpha = 0.7)
dx.set_ylim(2000,10000)
dx.set_yticks([2000,4000,6000,8000,10000])
dx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(e)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

ddx = fig.add_subplot(3,3,6)
dd = plt.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], dAdt_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_cross,cmap='RdBu_r',extend ='both')
plt.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], zlev[zlev1:zlev2], LWA_diversity_cros_com[1][zlev1:zlev2,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_cross, colors="k", alpha = 0.7)
ddx.set_ylim(2000,10000)
ddx.set_yticks([2000,4000,6000,8000,10000])
ddx.set_xticklabels(['-30','-20','-10','lon_c','+10','+20','+30'])
plt.title("(f)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.37,0.01,0.25])
cb = plt.colorbar(c, cax=cbar, ticks=[-2e-4,-1.5e-4,-1e-4,-0.5e-4, 0, 0.5e-4, 1e-4, 1.5e-4, 2e-4]) 
cb.set_ticklabels(['-2','-1.5','-1','0.5','0','0.5','1','1.5','2'])
cb.set_label('moist-induced LWA tendency ($10^{-4}$m/$s^2$)',fontsize=10)


lat_range=int(10/dlat)+1
lon_range=int(180/dlon)+1
t_range=6
duration = 6
maxlevel = LWA_Hov[2][:,:].max()
minlevel = LWA_Hov[2][:,:].min()  
levs_LWA_Hov = np.linspace(1e8, 2e8, 7)
maxlevel = dAdt_Hov[2][:,:].max()
minlevel = dAdt_Hov[2][:,:].min()  
levs_dAdt_Hov = np.linspace(-5e-5, 5e-5, 11)
        
ex = fig.add_subplot(3,3,7)
ex.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.7)
ex.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Hov[0][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
ex.set_ylabel('lag (days)',fontsize=12)
ex.set_yticks([0,3,6,9,12])
ex.set_yticklabels([-6,-3,0,3,6])
ex.set_xticks([-90,-60,-30,0,30,60,90])
ex.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(g)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

fx = fig.add_subplot(3,3,8)
fx.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.7)
f = fx.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Hov[2][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
fx.set_yticks([0,3,6,9,12])
fx.set_yticklabels([-6,-3,0,3,6])
fx.set_xticks([-90,-60,-30,0,30,60,90])
fx.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(h)", pad=5, fontdict={'family':'Times New Roman', 'size':12})

ffx = fig.add_subplot(3,3,9)
ffx.contour(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),LWA_Hov[1][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_LWA_Hov, colors="k", linewidths=0.7)
ff = ffx.contourf(lon[int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], np.arange(13),dAdt_Hov[1][int((2*duration+1)/2)-t_range:int((2*duration+1)/2)+t_range+1,int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1], levs_dAdt_Hov, cmap='RdBu_r',extend ='both')
ffx.set_yticks([0,3,6,9,12])
ffx.set_yticklabels([-6,-3,0,3,6])
ffx.set_xticks([-90,-60,-30,0,30,60,90])
ffx.set_xticklabels(['-90','-60','-30','lon_c','+30','+60','+90'])
plt.xlabel('relative longitude',fontsize=12)
plt.title("(i)", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cbar = fig.add_axes([0.93,0.1,0.01,0.25])
cb = plt.colorbar(f, cax=cbar, ticks=[-4e-5, -3e-5,-2e-5,-1e-5, 0 ,1e-5,2e-5,3e-5,4e-5]) 
cb.set_ticklabels(['-4.0','-3.0','-2.0','-1.0','0','1.0','2.0','3.0','4.0'])
cb.set_label('moist-induced LWA tendency ($10^{-5}$m/$s^2$)',fontsize=10)
plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/FigureS5.png",dpi=600)


#################################################################################################################
#%%
####### Figure 6 ##########
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_ridge_list2", "rb") as fp:
    dAdt_ridge_list= pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_ridge_list2", "rb") as fp:
    dTdt_ridge_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_dipole_list2", "rb") as fp:
    dAdt_dipole_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_dipole_list2", "rb") as fp:
    dTdt_dipole_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_trough_list2", "rb") as fp:
    dAdt_trough_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_trough_list2", "rb") as fp:
    dTdt_trough_list = pickle.load(fp) 

 
proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=[12,14])
ax = fig.add_subplot(4,1,1, projection=proj)
dT = ax.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], c=np.array(dTdt_ridge_list), transform=ccrs.PlateCarree(), vmin=0, vmax=2e-5, cmap='hot_r' ,s= 50, label="dTdt_moist")    
ax.set_title("(a) Moisture’s Thermodynamic Contributions for Ridge Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.5)
# ax.set_ylabel("dTdt_moist", fontsize=12)
ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
ax.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 


bx = fig.add_subplot(4,1,2, projection=proj)
dT = bx.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], c=np.array(dTdt_dipole_list), transform=ccrs.PlateCarree(), vmin=0, vmax=2e-5, cmap='hot_r', s=50, label='dTdt_moist')   
bx.set_title("(b) Moisture’s Thermodynamic Contributions for Dipole Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
bx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.5)
bx.coastlines()
bx.gridlines(linestyle="--", alpha=0.7)
bx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter) 
cbar = fig.add_axes([0.93,0.5,0.01,0.38])
cb = plt.colorbar(dT, cax=cbar, ticks=[0,0.5e-5,1e-5,1.5e-5,2e-5]) 
cb.set_ticklabels(['0','0.5','1','1.5','2'])
cb.set_label('moist-induced diabatic heating ($10^{-5}$K/s)',fontsize=10)


cx = fig.add_subplot(4,1,3, projection=proj)
cx.scatter(Blocking_diversity_peaking_lon[0][:],Blocking_diversity_peaking_lat[0][:], c=np.array(dAdt_ridge_list), transform=ccrs.PlateCarree(), vmin=-1e-4, vmax=1e-4, cmap='RdBu_r' ,s= 50, label="dAdt_moist")    
cx.set_title("(c) Moisture-induced Wave Activity Tendency for Ridge Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
cx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.5)
cx.coastlines()
cx.gridlines(linestyle="--", alpha=0.7)
cx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
cx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
cx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
cx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
cx.xaxis.set_major_formatter(lon_formatter)
cx.yaxis.set_major_formatter(lat_formatter) 


dx = fig.add_subplot(4,1,4, projection=proj)
dA = dx.scatter(Blocking_diversity_peaking_lon[2][:],Blocking_diversity_peaking_lat[2][:], c=np.array(dAdt_dipole_list), transform=ccrs.PlateCarree(), vmin=-1e-4, vmax=1e-4, cmap='RdBu_r', s=50, label='dAdt_moist')   
dx.set_title("(d) Moisture-induced Wave Activity Tendency for Dipole Blocks", pad=5, fontdict={'family':'Times New Roman', 'size':12})
dx.add_feature(cartopy.feature.LAND, facecolor='lightgray',alpha = 0.5)
dx.coastlines()
dx.gridlines(linestyle="--", alpha=0.7)
dx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
dx.set_extent([-180,180,0,90],crs=ccrs.PlateCarree())
dx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
dx.set_yticks([0,30,60,90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
dx.xaxis.set_major_formatter(lon_formatter)
dx.yaxis.set_major_formatter(lat_formatter) 
cbar = fig.add_axes([0.93,0.1,0.01,0.38])
cb = plt.colorbar(dA, cax=cbar, ticks=[-1.5e-4,-1e-4,-0.5e-4,0,0.5e-4, 1e-4,1.5e-4]) 
cb.set_ticklabels(['-1.5','-1','-0.5','0','0.5','1','1.5'])
cb.set_label('moist-induced LWA tendency ($10^{-4}$m/$s^2$)',fontsize=10)

# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/Figure6.png",dpi=600)

#################################################################################################################
#%%
#### Figure S2 ########
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_ridge_list2", "rb") as fp:
    dAdt_ridge_list= pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_ridge_list2", "rb") as fp:
    dTdt_ridge_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dAdt_dipole_list2", "rb") as fp:
    dAdt_dipole_list = pickle.load(fp) 
with open("/depot/wanglei/data/Reanalysis/MERRA2/Blocking_revised/dTdt_dipole_list2", "rb") as fp:
    dTdt_dipole_list = pickle.load(fp) 
# Data
categories = ['Ridge blocks', 'Dipole blocks']
values1 = dTdt_ridge_list  # Values for the first category
values2 = dTdt_dipole_list # Values for the second category
values3 = dAdt_ridge_list  # Values for the first category
values4 = dAdt_dipole_list # Values for the second category

# Define colors for the boxes
box_colors = ['salmon','skyblue']

# Plotting
fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1)
# Plot box plot
boxplot = ax.boxplot([values1, values2], positions=[0, 1], widths=0.5, showfliers=False, whis=(5, 95), patch_artist = True,medianprops=dict(color='black'))
ax.set_ylim(0,3e-5)
ax.set_xticks([0, 1])
ax.set_xticklabels(categories)
ax.set_yticks([0,0.5e-5,1e-5,1.5e-5,2e-5,2.5e-5,3e-5])
ax.set_yticklabels(['0','0.5','1','1.5','2','2.5','3'])
ax.set_ylabel('dTdt_moist ($10^{-5}$K/s)',fontsize=12)
ax.set_title('(a) Distribution of moisture-induced diabatic heating',fontsize=12)
for box, color in zip(boxplot['boxes'], box_colors):
    box.set_facecolor(color)

bx = fig.add_subplot(1,2,2)
# Plot box plot
boxplot = bx.boxplot([values3, values4], positions=[0, 1], widths=0.5, showfliers=False, whis=(5, 95), patch_artist = True,medianprops=dict(color='black'))
bx.hlines(0, -0.5, 1.5, linestyles='solid',color='k')
bx.set_ylim(-0.00015,0.00015)
bx.set_xticks([0, 1])
bx.set_xticklabels(categories)
bx.set_yticks([-1.5e-4,-1e-4,-0.5e-4,0,0.5e-4,1e-4,1.5e-4])
bx.set_yticklabels(['-1.5','-1','-0.5','0','0.5','1','1.5',])
bx.set_ylabel('dAdt_moist ($10^{-4}$m/$s^2$)',fontsize=12)
bx.set_title('(b) Distribution of moisture-induced wave activity tendency',fontsize=12)
for box, color in zip(boxplot['boxes'], box_colors):
    box.set_facecolor(color)
plt.tight_layout()
# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/FigureS1.png",dpi=600)

#################################################################################################################
#%%
#### Figure S1 ########
categories = ['Ridge Blocks', 'Dipole Blocks']
values1 = np.array(Blocking_diversity_duration[0])  # Values for the first category
values2 = np.array(Blocking_diversity_duration[2]) # Values for the second category

# Define colors for the boxes
box_colors = ['salmon','skyblue']

# Plotting
fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1)
# Plot box plot
boxplot = ax.boxplot([values1, values2], positions=[0,1], widths=0.5, showfliers=False, whis=(5, 95), patch_artist = True,medianprops=dict(color='black'))
ax.set_ylim(0,15)
ax.set_xticks([0, 1])
ax.set_xticklabels(categories)
ax.set_yticks([0,5,10,15,20])
ax.set_yticklabels(['0','5','10','15','20'])
ax.set_ylabel('duration (days)',fontsize=12)
ax.set_title('Duration of two blocks',fontsize=12)
for box, color in zip(boxplot['boxes'], box_colors):
    box.set_facecolor(color)

# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/duration.png",dpi=600)


##### To see the transiation of different types within a block #####
##### e.g. In dipoles, how many days are dipole and how many days are ridges? ######
Blocking_diversity_transition = []
Blocking_diversity_ridge_day = []
Blocking_diversity_trough_day = []
Blocking_diversity_dipole_day = []

Blocking_diversity_ridge_day_frac = []
Blocking_diversity_trough_day_frac = []
Blocking_diversity_dipole_day_frac = []

lon_range=int(30/dlon)+1

for i in np.array([0,1,2]):
    transition_total = []
    ridge_day_total = [];    trough_day_total = [];   dipole_day_total = []
    ridge_day_fraction = []; trough_day_fraction =[]; dipole_day_fraction = [] 
    for n in np.arange(len(Blocking_diversity_date[i])):
        transition = []
        ridge_day = 0; trough_day = 0; dipole_day = 0
        for t in np.arange(len(Blocking_diversity_date[i][n])):
                
            date_index = Date.index(Blocking_diversity_date[i][n][t])
            lon_index = np.squeeze(np.array(np.where( lon[:]==Blocking_diversity_lon[i][n][t])))
            lat_index = np.squeeze(np.array(np.where( lat[:]==Blocking_diversity_lat[i][n][t])))
        
            ### date LWA_AC ###
            file_LWA_AC = Dataset(path_LWA_AC[date_index],'r')
            LWA_AC  = file_LWA_AC.variables['LWA_Z500'][0,0,180:,:]
            file_LWA_AC.close()
            
            ### date LWA_C ###
            file_LWA_C = Dataset(path_LWA_C[date_index],'r')
            LWA_C  = file_LWA_C.variables['LWA_Z500'][0,0,180:,:]
            file_LWA_C.close()

            LWA_AC = np.roll(LWA_AC, int(nlon/2)-lon_index, axis=1)
            LWA_C = np.roll(LWA_C,   int(nlon/2)-lon_index, axis=1)
            WE = np.roll(Blocking_diversity_label[i][n][t], int(nlon/2)-lon_index, axis=1)
            lon_roll = np.roll(lon,   int(nlon/2)-lon_index)
            
            LWA_AC = LWA_AC[:, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
            LWA_C = LWA_C[  :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
            WE = WE[        :, int(nlon/2)-int(lon_range/2):int(nlon/2)+int(lon_range/2)+1]
            
            LWA_AC_d = np.zeros((nlat_NH, lon_range))
            LWA_C_d = np.zeros((nlat_NH, lon_range))
            LWA_AC_d[WE == True]  = LWA_AC[WE== True]
            LWA_C_d[WE == True]  =  LWA_C[WE == True]
            
            LWA_AC_sum = LWA_AC_d.sum()
            LWA_C_sum = LWA_C_d.sum()
            
            if LWA_AC_sum > 10 * LWA_C_sum:
                transition.append(0)
                ridge_day+=1
            elif LWA_C_sum > 2 * LWA_AC_sum:
                transition.append(1)
                trough_day+=1
            else:
                transition.append(2)
                dipole_day+=1
        
        transition_total.append(transition)
        ridge_day_total.append(ridge_day);   ridge_day_fraction.append(ridge_day/Blocking_diversity_duration[i][n])
        trough_day_total.append(trough_day); trough_day_fraction.append(trough_day/Blocking_diversity_duration[i][n])
        dipole_day_total.append(dipole_day); dipole_day_fraction.append(dipole_day/Blocking_diversity_duration[i][n])
        
        print(n)

    Blocking_diversity_ridge_day.append(ridge_day_total)
    Blocking_diversity_trough_day.append(trough_day_total)
    Blocking_diversity_dipole_day.append(dipole_day_total)
    Blocking_diversity_ridge_day_frac.append(ridge_day_fraction)
    Blocking_diversity_trough_day_frac.append(trough_day_fraction)
    Blocking_diversity_dipole_day_frac.append(dipole_day_fraction)
    Blocking_diversity_transition.append(transition_total)

#%%
### Make some bar plots for the transition of blocks ###
dipole_day_frac = pd.DataFrame(Blocking_diversity_dipole_day_frac[2])
ridge_day_frac = pd.DataFrame(Blocking_diversity_ridge_day_frac[2])
trough_day_frac = pd.DataFrame(Blocking_diversity_trough_day_frac[2])

dipole_day_frac.hist(bins=21,range=(0,1))
plt.xlim(0,1)
plt.title('dipole day fraction in dipole blocks') 
plt.xlabel('fraction') 
plt.ylabel('number')

#%%
### Figure S7 ###
# Data for the plot
labels = ['dipole blocks', 'ridge blocks']
section1 = [np.mean(Blocking_diversity_ridge_day_frac[2]), np.mean(Blocking_diversity_ridge_day_frac[0])]
section2 = [np.mean(Blocking_diversity_dipole_day_frac[2]), np.mean(Blocking_diversity_dipole_day_frac[0])]
section3 = [np.mean(Blocking_diversity_trough_day_frac[2]), np.mean(Blocking_diversity_trough_day_frac[0])]

# Bar width
barWidth = 0.3
x = np.array([0,1])
# Stacking the sections
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

colors = ['lightpink','lightsteelblue','navajowhite']

ax.bar(x, section1, barWidth, label='ridge', color=colors[0])
ax.bar(x, section2, barWidth, bottom=section1, label='dipole', color=colors[1])
ax.bar(x, section3, barWidth, bottom=np.add(section1, section2), label='trough', color=colors[2])

# Adjust x-ticks to be centered on the bars
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Adding labels and title
ax.set_ylabel('time fraction (days / total days)')
ax.set_title('Wave structures during blocking evolution')

# Customizing the legend (removing the box and setting position)
plt.legend(frameon=False)

# plt.savefig("/home/liu3315/Research/Blocking_Diversity/Fig_revised/Figure S7.png",dpi=600)

# %%
