#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:44:45 2025

@author: ariel2
"""

import os
import re
import numpy as np
import pandas as pd
import optimal_functions as of
import matplotlib.pyplot as plt
import imageio.v2 as imageio

os.makedirs("frames", exist_ok=True)
#retriev model data
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
mod_id='sm_3'
top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
os.chdir(f'/data/optimal/mod_files/{mod_id}')
os.makedirs("frames", exist_ok=True)

# split ssm files into stress periods
input_file_name=f'/data/optimal/mod_files/{mod_id}/{mod_id}.ssm'
# Run the splitting function
of.split_file_by_stress_period(input_file_name)

#Defining stress period data

sp_sealevel=[ -27.3,  -14. ,  -23.2,  -49.6,  -80.5,  -83.2,  -70.1,  -65.6,
    -66.5,  -74. ,  -87.3, -100.7, -109. , -104. ,  -99.5, -101.3,
   -102.2,  -93.3,  -54.4,   -5.9,   -3.3,  -16.6,  -40.4,  -46.9,
    -41.6,  -33.6,  -38.3,  -46.6,  -48.1,  -47.8,  -33. ,  -41.6,
    -55. ,  -81.7,  -91.8,  -82.3,  -74.3,  -73.4,  -78.7,  -82. ,
    -87.3,  -90.6,  -90. ,  -92.4, -103.1, -115. , -109. ,  -70.7,
    -31.8,  -10.4,    0. , 0.]
for sp in range(1,len(sp_sealevel)):
   
    sl=sp_sealevel[sp-1] #get the corresponding sealevel from the list
    
    #Creating dataframe
    var_names=['i','j','k','conc','bc']
    filename=f"/data/optimal/mod_files/{mod_id}/ssm_sp_{sp}.txt"
    df_ssm = pd.read_csv(filename,skiprows=1,names=var_names,sep='\s+')
    ni=100*df_ssm['k'].values[:-1].max()
    
    df_ssm_chd=df_ssm[df_ssm['bc']==1] # CHD boundary cells
    
    df_ssm_ghb=df_ssm[df_ssm['bc']==5] # GHB boundary cells
    
    fig, ax1=plt.subplots(figsize=(8,3))
    
    ax1.scatter(100*df_ssm['k'].values[:-1],-10*df_ssm['i'].values[:-1],c=df_ssm['conc'].values[:-1],
                cmap='jet',s=20)
    ax1.scatter(100*df_ssm_chd['k'].values,-10*df_ssm_chd['i'].values,marker='o',facecolors=None, edgecolors='g', s=2)
    ax1.scatter(100*df_ssm_ghb['k'].values,-10*df_ssm_ghb['i'].values,marker='v',facecolors=None, edgecolors='y', s=2)
    ax1.plot(np.arange(1,ni),(sl-top_elev)*np.ones(int(ni-1)),'b--',linewidth=1,label='Sealevel')
    ax1.set_title(f'Model BC at str. per {sp}, rsl: {sl} m')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_ylim(-1000,0)
    plt.tight_layout()
    #fig.savefig(f"frames/frame_{sp:03d}.png", dpi=150)
    plt.show()
    plt.close(fig)
    
# # Now create the GIF
# with imageio.get_writer(f'{mod_id}_bc_timelapse.gif', mode='I', duration=0.5) as writer:
#     for sp in range(len(sp_sealevel)):
#         filename = f"frames/frame_{sp:03d}.png"
#         image = imageio.imread(filename)
#         writer.append_data(image)