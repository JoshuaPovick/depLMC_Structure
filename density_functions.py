#!/usr/bin/env python
# coding: utf-8

# In[1]:


#########################
### Data Manipulation ###
#########################

import numpy as np

def brtfntlmc_bins(apogee_data):
    '''
    Bin data by APOGEE Fields and Brt/Fnt RGB
    
    apogee_data: fits file of apogee data
    '''
    
    field_labels = ['30Dor','LMC1','LMC2','LMC3','LMC4','LMC5','LMC6','LMC7','LMC8','LMC9','LMC10','LMC11',
                    'LMC12','LMC13','LMC14','LMC15','LMC16','LMC17']
    
    binned_data = dict()
    
    #find faint and bright stars and bin apogee
    fnts = np.where(apogee_data['TARGET_NAME']=='FntRGB')
    brts = np.where(apogee_data['TARGET_NAME']=='BrtRGB')
    
    fnts_apogee = apogee_data[fnts]
    brts_apogee = apogee_data[brts]
    
    #find different fields in the fnt and brt bins and add to dictionary
    for i in range(len(field_labels)):
        fnt_key = '{}'.format(field_labels[i]) + '_fnt'
        fnt_val = np.where(fnts_apogee['FIELD'] == field_labels[i])
        binned_data[fnt_key] = fnt_val
        
        brt_key = '{}'.format(field_labels[i]) + '_brt'
        brt_val = np.where(brts_apogee['FIELD'] == field_labels[i])
        binned_data[brt_key] = brt_val
    
    return binned_data

def mad(dat):
    '''
    Calculate the median absolute deviation of 1d array-like object dat
    '''
    return np.median(np.absolute(dat - np.median(dat)))


# In[ ]:


#############################
### Field Mass Calcultion ###
#############################

###Calculate APOGEE field mass with PARSEC isochrones
import astropy
from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np
from scipy.interpolate import interp1d

def findclosestparsec(age,metallicity,parsec_ascii_table):
    
    #find closest parsec isochrone z fraction
    parsec_mets = np.unique(np.asarray((np.log10(parsec_ascii_table['Zini']/0.02))))
    met_index = np.absolute(parsec_mets-metallicity*np.ones(len(parsec_mets))).argmin()
    z_iso = round(0.02*(10**parsec_mets[met_index]),10)
    
    #find closest parsec isochrone age
    age = np.log10(age*(10**9))
    parsec_ages = np.unique(np.asarray(parsec_ascii_table['logAge']))
    age_index = np.absolute(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
    age_iso = parsec_ages[age_index]
    
    #pick out isochrone
    #single_iso = np.where((parsec_ascii_table['Zini']==z_iso)&
    #                      (parsec_ascii_table['logAge']==age_iso)&
    #                      (parsec_ascii_table['Hmag']<=maxabsH))
    
    return age_iso, z_iso

def fieldmass(age_iso,z_iso,maxabsH,minabsH,selectfunc,number,parsec_ascii_table):
    
    single_iso = np.where((parsec_ascii_table['Zini']==z_iso)&
                          (parsec_ascii_table['logAge']==age_iso)&
                          (parsec_ascii_table['Hmag']<=maxabsH))
    
    #upto = np.where(parsec_ascii_table[single_iso]['Hmag']==np.min(parsec_ascii_table[single_iso]['Hmag']))
    #new_parsec = parsec_ascii_table[single_iso][0:int(np.squeeze(upto))]
    
    x1 = parsec_ascii_table[single_iso]['Hmag']
    y1 = parsec_ascii_table[single_iso]['int_IMF']
    
    points = zip(x1, y1)
    
    points = sorted(points, key=lambda point: point[0])
    
    x2, y2 = zip(*points)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    
    Hcut = np.where(x2<-2)
    x2 = x2[Hcut]
    y2 = y2[Hcut]
    
    inter_iso = interp1d(x2,y2,kind='cubic',bounds_error=False,fill_value='extrapolate',assume_sorted=False)
    
    #inter_iso = interp1d(parsec_ascii_table[single_iso]['Hmag'][::-1],
                         #parsec_ascii_table[single_iso]['int_IMF'][::-1],kind='cubic',
                         #bounds_error=False,fill_value='extrapolate',assume_sorted=False) 
                          
                          #fill_value=(np.nan,np.nan)
    
#     inter_iso = interp1d(new_parsec['Hmag'][::-1],new_parsec['int_IMF'][::-1],kind='cubic',bounds_error=False,
#                          fill_value='extrapolate',assume_sorted=False) #fill_value=(np.nan,np.nan)
    
    diff_IMF = np.absolute(inter_iso(maxabsH)-inter_iso(minabsH))
    
    return (selectfunc*number)/diff_IMF


# In[ ]:


################
### Geometry ###
################

def LMCdisk_cart(ra, dec):
    
    '''
    Calculate the position of stars in the LMC disk plane with 
    center at the LMC center in cartesian coordinates (x, y).
    This also calculates the distance to the individual stars.
    
    This follows van der Marel and Cioni 2001 
    
    Input
    - ra: right ascension of stars
    - dec: declination of stars
    
    Output
    - x_m: x coordinate
    - y_m: y coordinate
    - dis: distance to LMC star
    '''
    alph0 = np.radians(82.25) #right ascension of center of LMC
    delt0 = np.radians(-69.50) #declination of center of LMC
    pa = np.radians(149.23+90.00) #146.37 #position angle of line of nodes
    io = np.radians(25.86) #27.81 #inclination of LMC disk
    d0 = 49.90 #distance to center of LMC
    
    #convert to radians
    ra = np.radians(ra)
    dec = np.radians(dec)
    sd = np.sin(delt0)
    cd = np.cos(delt0)
    
    cr = cd*np.cos(dec)*np.cos(ra-alph0)+sd*np.sin(dec)
    srcp = -np.cos(dec)*np.sin(ra-alph0)
    srsp = cd*np.sin(dec) - sd*np.cos(dec)*np.cos(ra-alph0)
    dis = d0*np.cos(io)/(np.cos(io)*cr - np.sin(io)*np.cos(pa)*srsp + np.sin(io)*np.sin(pa)*srcp)
    
    x_m = dis*srcp
    y_m = dis*(np.cos(io)*srsp + np.sin(io)*cr) - d0*np.sin(io)
    
    return x_m, y_m, dis

import astropy
from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np
from scipy.spatial import ConvexHull

def apogee_field_area(field,data):
    
    """
    This calculates the areal extent of an APOGEE field (kpc^2)
    - field: string of name of field to get area
    - data: table of data for field
    """
    
    fld = np.where(data['FIELD']==field)
    
    ###calculate the centroid
    x_m0, y_m0, dist = LMCdisk_cart(data[fld]['RA'],data[fld]['DEC'])
    
    points = []
    for j in range(len(np.squeeze(fld))):
        points.append([x_m0[j],y_m0[j]])
    points = np.asarray(points)
    
    #find exterior points and area using Convex Hull algorithm 
    hull = ConvexHull(points)
    
    return hull.volume


# In[ ]:


#############
### Other ###
#############

def sal(MH,aM):
    '''
    Calculate the Salaris correction to the overall metallicity
    '''
    return MH + np.log(0.638*(10**(aM))+0.362)

#calculate absolute mag
def absmag(magnitude,distance):
    '''
    - magnitude: apparent magnitude of star
    - distance: distance to star in kpc
    Calculate the absolute magnitude of star
    '''
    absm = []
    absm.append(magnitude-5.0*np.log10(distance*1000)+5.0)
    absm = np.squeeze(np.array(absm))
    return absm
