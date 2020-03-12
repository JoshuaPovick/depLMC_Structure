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

def findclosestparsec(parsec_ascii_table,age,feh):
    '''
    Find closest PARSEC isochrone and return the age and z fraction
    
    parsec_ascii_table: ascii table of isochrone to use (label = 3, RGB stars)
    age: age of field to use
    feh: metallicity 
    '''
    
    #find closest parsec isochrone z fraction
    parsec_mets = np.unique(np.asarray((np.log10(parsec_ascii_table['Zini']/0.02))))
    met_index = np.absolute(parsec_mets-feh*np.ones(len(parsec_mets))).argmin()
    z_iso = round(0.02*(10**parsec_mets[met_index]),10)
    
    #find closest parsec isochrone age
    age = np.log10(age*(10**9))
    parsec_ages = np.unique(np.asarray(parsec_ascii_table['logAge']))
    age_index = np.absolute(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
    age_iso = parsec_ages[age_index]
    
    return age_iso, z_iso

def fieldmass(age_iso,z_iso,absH,sf_brt,sf_fnt,parsec_ascii_table):
    
    '''
    Calculate Mass of APOGEE Field with PARSEC isochrones:
    
      SF_brt x N_brt + SF_fnt x N_fnt
    ----------------------------------- = field mass in mSol
    intIMF(minH_brt) - intIMF(maxH_fnt)
    
    age_iso: Age of PARSEC isochrone to use
    z_iso: Z fraction of PARSEC isochrone to use
    
    absH: arraylike object that contains all the Hmags for the whole field
    
    sf_brt: arraylike object that contains selection function for brt stars
    sf_fnt: arraylike object that contains selection function for fnt stars
    
    parsec_ascii_table: ascii table of isochrone to use (label = 3, RGB stars)
    
    '''
    
    single_iso = np.where((parsec_ascii_table['Zini']==z_iso)&
                          (parsec_ascii_table['logAge']==age_iso))#&
                          #(parsec_ascii_table['Hmag']<=np.max(absH)))
    
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
    
    diff_IMF = np.absolute(inter_iso(np.max(absH))-inter_iso(np.min(absH)))
    
    if len(sf_fnt) == 0:
        numerator = sf_brt[0]*len(sf_brt) 
        
    else:
        numerator = sf_brt[0]*len(sf_brt) + sf_fnt[0]*len(sf_fnt)
    
    return numerator/diff_IMF


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


#################
### STAR AGES ###
#################

### Age model
def parsec_age(w,x,y,z):
    p=[28.90025823,-0.8303683,3.28088688,-0.08771859,-7.48008086,-0.66424502,\
       0.04407313,0.13976222,0.74247359]
    return p[0]+p[1]*w+p[2]*x+p[3]*y+p[4]*z+p[5]*np.multiply(x,z)+p[6]*np.multiply(y,z)+p[7]*(x**2)+p[8]*(z**2)

def noisydata(lgT,lgTERR,ks,ksERR,mh,mhERR,cfe,cfeERR,nfe,nfeERR,feh,fehERR,lgg,lggERR):
    carbon = 0.28115244582676185 # derived in initial age calc
    nitrogen = 0.06901474154376043 # derived in initial age calc
    Tnoise = np.random.normal(0, 0.434*(lgTERR/lgT)) #logTeff
    Knoise = np.random.normal(0, ksERR) #Ks
    MHnoise = np.random.normal(0, mhERR) #[M/H]
    
    cm = cfe + feh - mh
    nm = nfe + feh - mh
    CMERR = np.sqrt((cfeERR)**2+(fehERR)**2+(mhERR)**2)
    NMERR = np.sqrt((nfeERR)**2+(fehERR)**2+(mhERR)**2)
    
    expcarERR = 10**(cm)*np.log(10)*CMERR
    expnitERR = 10**(nm)*np.log(10)*NMERR
    
    xcarb = carbon*10**(cm)
    xnitr = nitrogen*10**(nm)
    fac = (xcarb+xnitr)/(carbon+nitrogen) #factor from Salaris correction
    facERR = np.sqrt((expcarERR)**2+(expnitERR)**2)/(carbon+nitrogen)
    
    facnoise = np.random.normal(0, np.absolute(0.434*(facERR/fac)))
    
    lggnoise = np.random.normal(0, lggERR) #logg
    Tnew = lgT + ((-1)**np.random.randint(2))*Tnoise
    Knew = ks + ((-1)**np.random.randint(2))*Knoise
    MHnew = mh + ((-1)**np.random.randint(2))*MHnoise
    facnew = fac + ((-1)**np.random.randint(2))*facnoise
    lggnew = lgg + ((-1)**np.random.randint(2))*lggnoise
    return Tnew, Knew, MHnew, facnew, lggnew

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
