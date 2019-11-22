#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import astropy
from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np
from scipy.spatial import ConvexHull

def apogee_field_area(field,data):
    
    """
    This calculates the areal extent of an APOGEE field
    - field: string of name of field to get area
    - data: table of data for field
    """
    
    fld = np.where(data['FIELD']==field)
    
    ###calculate the centroid
    
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
    
    x_m0, y_m0, dist = LMCdisk_cart(data[fld]['RA'],data[fld]['DEC'])
    
    points = []
    for j in range(len(np.squeeze(fld))):
        points.append([x_m0[j],y_m0[j]])
    points = np.asarray(points)
    
    #find exterior points and area using Convex Hull algorithm 
    hull = ConvexHull(points)
    
    return hull.volume
