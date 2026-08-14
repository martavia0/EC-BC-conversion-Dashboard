# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:18:46 2026

@author: marta
"""

import pandas as pd
import numpy as np
# import xarray as xr
import h5py

path_model = r"C:\Users\marta\Documents\SMASH\SMASH_EC\Results\STAN_modelling\Models_aftercoauth\idata_EC_BC_model6.h5"

path_model = r"C:\Users\marta\Documents\SMASH\SMASH_EC\Results\STAN_modelling\Models_aftercoauth\idata_EC_BC_model6.h5"
path_out = r"C:\Users\marta\Documents\GitHub\Dashboard\EC-BC-conversion-Dashboard/idata_EC_BC_model6_slim.h5"

vars_needed = [
    "slope",
    "protocol",
    "ins_ec",
    "ins_bc",
    "size_ec_1",
    "size_ec_10",
    "size_bc_10",
    "sigma",
]

with h5py.File(path_model, 'r') as f_in, h5py.File(path_out, 'w') as f_out:
    grp_in = f_in['posterior']
    grp_out = f_out.create_group('posterior')

    for var in vars_needed:
        # copy the dataset itself
        grp_in.copy(var, grp_out)

        # copy over any per-variable attrs (e.g. dims, units) if present
        if var in grp_in and hasattr(grp_in[var], 'attrs'):
            for key, val in grp_in[var].attrs.items():
                grp_out[var].attrs[key] = val

    # copy group-level attrs (chain/draw coordinates etc. if stored as attrs)
    for key, val in grp_in.attrs.items():
        grp_out.attrs[key] = val