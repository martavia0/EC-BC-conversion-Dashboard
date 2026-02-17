# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 17:35:53 2026

@author: marta
"""

import pandas as pd
import numpy as np
import h5py

def ecbc_conversion(orig, datetime, ec_bc, prot, instrEC, instrBC, scec1, scec10, scbc10, scbct):
    
    orig = np.asarray(orig, dtype=float)
    datetime = np.asarray(datetime)
    #The file is from the 10th of Dec
    with h5py.File(r"C:\Users\marta\Documents\SMASH\SMASH_EC\Results\STAN_modelling\Linear\ECBC_MLR_0114.h5", 'r') as f:
        # print(list(f.keys()))       
        slope = np.array(f['slope'])
        protocol = np.array(f['protocol'])
        instr_ec = np.array(f['ins_ec'])
        instr_bc = np.array(f['ins_bc'])
        size_ec_1 = np.array(f['size_ec_1'])
        size_ec_10 = np.array(f['size_ec_10'])
        size_bc_10 = np.array(f['size_bc_10'])
        size_bc_tsp = np.array(f['size_bc_tsp'])
        sigma = np.array(f['sigma'])

    # We keep it in log space since the matrix multiplication needs to be done through log terms
    # Otherwise, we are summing all contributions of the conversion instead of multiplying them.
    mlr = pd.DataFrame({'Slope': slope, 'Protocol': protocol,
                        'Instr_EC': instr_ec, 'Instr_BC': instr_bc,
                        'Size_EC_1': size_ec_1, 'Size_EC_10': size_ec_10,  
                        'Size_BC_10': size_bc_10, 'Size_BC_tsp': size_bc_tsp,  
                        'Sigma': sigma})
    mlr_matrix = mlr.iloc[2000:,:-1].T.to_numpy()

    binary_matrix = np.array([ec_bc, prot, instrEC, instrBC, scec1, scec10, scbc10, scbct]).T
    log_orig = np.log10(orig)
    mlr_matrix[0, :] *= -1   # des-invert ONLY first column (The slope is to be applied to BC, )
    # print("Dim binary matrix: ", binary_matrix.shape)
    # print("Dim MLR_matrix: ", mlr_matrix.shape)

    conversion_matrix = np.dot(binary_matrix, mlr_matrix)
    
    converted = 10**(log_orig[:, None] - conversion_matrix) #The minus is because we need to invert the conversion matrix
    converted = pd.DataFrame(converted)
    
    conv_df = pd.DataFrame()
    conv_df['Datetime']=pd.to_datetime(datetime, yearfirst=True)   
    conv_df['original']=orig
    conv_df['converted_EC_BC'] = np.median(converted, axis=1)
 
    return converted, conv_df
#%%
orig=[1.2, 1.3, 1.25]
dates = ["01/01/2020", "02/01/2020", "02/01/2020"]
ec_bc =  np.full(len(orig),1)
protocol = np.full(len(orig),1)
instr_ec = np.full(len(orig),0)
instr_bc =  np.full(len(orig),0)
scec1= np.full(len(orig),1)
scec10= np.full(len(orig),0)
scbc10= np.full(len(orig),0)
scbctsp=  np.full(len(orig),0)

a,b = ecbc_conversion(orig, dates, ec_bc, protocol, instr_ec, instr_bc, scec1, scec10, scbc10, scbctsp)


