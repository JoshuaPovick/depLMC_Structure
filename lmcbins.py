#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np

def brtfntlmc_bins(apogee_data):
    '''
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

