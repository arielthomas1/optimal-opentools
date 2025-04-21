#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:31:02 2025

@author: ariel2
"""

import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
work_dir='/home/ariel2/Projects/optimal_mod_runs/'
os.chdir(work_dir)
from matplotlib import colors
#%%
mod_name='sm_7'
par='conc'

# Load the NetCDF file
file_path = os.path.join(work_dir,mod_name, 'results','_netcdf','_{}_{}.nc'.format(par,mod_name))
ds = xr.open_dataset(file_path)

# Print basic info about the dataset
print(ds)

# Print all variable names
print("\nVariables in dataset:")
print(list(ds.data_vars))

# View coordinates
print("\nCoordinates:")
print(ds.coords)

# If you want a quick peek at the data
print("\nSample of one variable:")
print(ds[list(ds.data_vars)[0]])

var_name = 'conc'  # Change this to your actual variable name
conc_arr = ds[var_name].values
conc_arr[conc_arr > 1000.] = np.nan
# Slice the data at time=0 and y=50 to get Z vs X
ts=0
salinity_arr = conc_arr[ts, :, 0, :]

# Make the plot
plt.figure(figsize=(10, 6))
# Transpose to get Z on the y-axis and X on the x-axis (optional based on how your data is structured)
plt.imshow(salinity_arr,
           aspect='auto', 
           cmap='viridis')
plt.colorbar(label='Value')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('2D Cross-section (Z vs X)')
plt.tight_layout()
plt.savefig("cross_section.png", dpi=300)
plt.show()
