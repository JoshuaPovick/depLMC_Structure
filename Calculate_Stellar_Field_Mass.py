# coding: utf-8


###Calculate APOGEE field mass with PARSEC isochrones
import numpy as np
import astropy
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.interpolate import interp1d

def apogee_field_mass(age,metallicity,maxabsH,minabsH,selectfunc,number,parsec_path):
	"""
	This is used to calculate the mass of stars in a field
	- age: age of stars to use to pick out isochrone
	- metallicity: metallicity of stars to use to pick out isochrone
	- maxabsH: max value of H for the field
	- minabsH: min value of H for the field
	- selectfunc: selection for the field
	- number: number of stars observed in the field
	- parsec_path: dat file path for PARSEC isochrones
	"""

	###Load in isochrones from PARSEC
	parsecall = ascii.read(parsec_path, format='basic', delimiter='\s')
	rgb = np.where(parsecall['label']==3)
	parsec = parsecall[rgb]

	#find closest PARSEC isochrone z fraction
	parsec_mets = np.unique(np.asarray((np.log10(parsec['Zini']/0.02))))
	met_index = np.abs(parsec_mets-metallicity*np.ones(len(parsec_mets))).argmin()
	z_iso = 0.02*(10**parsec_mets[met_index])	

	#find closest PARSEC isochrone age
	age = np.log10(age*(10**9))
	parsec_ages = np.unique(np.asarray(parsec['logAge']))
    
	age_index = np.abs(parsec_ages-age*np.ones(len(parsec_ages))).argmin()
	age_iso = parsec_ages[age_index]

	###interpolate and find mass from int_IMF
	single_iso = np.where((parsec['Zini']==z_iso)&(parsec['logAge']==age_iso)&(parsec['Hmag']<=maxabsH))

	upto = np.where(parsec[single_iso]['Hmag']==min(parsec[single_iso]['Hmag']))
	new_parsec = parsec[single_iso][0:int(np.squeeze(upto))]

	inter_iso = interp1d(new_parsec['Hmag'][::-1],new_parsec['int_IMF'][::-1],kind='cubic',bounds_error=False,fill_value='extrapolate',assume_sorted=False)

	diff_IMF = np.abs(inter_iso(maxabsH)-inter_iso(minabsH))

	return (selectfunc*number)/diff_IMF
