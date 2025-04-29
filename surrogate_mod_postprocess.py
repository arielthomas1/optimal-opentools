#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:16:48 2025

@author: ariel2
"""

"""
    Postprocessing of model output via imod-WQ runs which involves combining the partitioned output files into
    the overall model grid. The files will be stored for each stress period individually in a newly created folder
    tree. For each stress period an overall .netcdf file will be created with the x, y, z and time dimensions storing
    heads, fluxes and concentrations.
"""

#   import libraries
import os
import sys
import _tool_plotting as tlz_tp
import numpy as np
import flopy.utils.binaryfile as bf
import xarray as xr
import re
#%%
#   the indexes are defined from the job script
a = #int((sys.argv[1]))

sp_sealevel=[ -27.3,  -35.9,  -83.2,  -65.3,  -87.3, -108.7, -101.3,  -77.2,
         -3.3,  -47.2,  -33.6,  -48.4,  -33. ,  -68.6,  -82.3,  -76.9,
        -87.3,  -90.6, -115. ,  -49. ,    0. ]

perlen = (365.25*9999)*np.ones_like(sp_sealevel) #an array filled with values of the length of each stress period in days.
nper = len(sp_sealevel) # number of stress periods
total_sim_time=sum(perlen)
timprs_lst = list(np.linspace(1,total_sim_time,nper,endpoint=True,dtype=int))
time_start = timprs_lst[0] # int((sys.argv[2]))
time_end = timprs_lst[-1] #int((sys.argv[2]))

#   define the model name
model_name = 'sm_7'
#   define grid dimensions for row, col (dx and dy) and lay thickness (dz)
dx = dy = 100
dz = 10

#   define the model stress periods, names etc.

sp_names = ['sm_7'] * len(sp_sealevel) # just set it to match the csv names
sp_time_dur =[1e05] * len(sp_sealevel)  # duriation of each stress period - in years
#sp_sea_level = [-130., -80., -20., 0.]  # sea level for each stress period

#   create the folder structure, first the main directory
out_dir = r'/home/ariel2/Projects/optimal_mod_runs'
#out_dir = r'g:\_NZ_Canterbury\Models'
main_dir = os.path.join(out_dir, model_name)   # overall parent directory
model_name_sp = sp_names[a]
sp_dir = os.path.join(main_dir, model_name_sp)
#%%
#   define the name of the mini stress period
if time_start == 1:
    time_start_str = '0000000'
elif 10000 <= time_start < 100000:
    time_start_str = '00' + str(time_start)
elif 100000 <= time_start < 1000000:
    time_start_str = '0' + str(time_start)
#   do the same for the end time
time_end = time_start + 10000
if 10000 <= time_end < 100000:
    time_end_str = '00' + str(time_end)
elif 100000 <= time_end < 1000000:
    time_end_str = '0' + str(time_end)

#   define the modelname and folder for the mini stress period
mini_modelname = 'SP_' + time_start_str + '_to_' + time_end_str
mini_sp_dir = os.path.join(sp_dir, mini_modelname)
#%%
#   create the results directory if it doesnt exist
main_res_dir = os.path.join(main_dir, 'results')
os.makedirs(main_res_dir, exist_ok = True)

#   load the ibound_arr (if it doesnt exist)
ibound_arr_dir = os.path.join(main_dir, 'ibound_arr.npy')
ibound_arr = np.load(ibound_arr_dir, allow_pickle = True)
#   get the nlay, nrow and ncol values
nlay, nrow, ncol = ibound_arr.shape[0], ibound_arr.shape[1], ibound_arr.shape[2]

#   create the main output folders
netcdf_dir = os.path.join(main_res_dir, '_netcdf', model_name_sp)
os.makedirs(netcdf_dir, exist_ok=True)
#%%
#   Extract the salinity concentrations and heads from the partition list and UCN files
#   find the list.p0000 file
num_cores=12
partition_idxs_lst = []
str_start = 'p000 :  '
str_end = 'p011 :  '
if os.path.isfile(os.path.join(main_dir, model_name + '.list.p000')):
    with open(os.path.join(sp_dir + '.list.p000'), 'r') as listfile:
        lines = listfile.readlines()
        #   loop through a list of core numbers
        for i in range(num_cores):
            if i < 10:
                str_i = '0' + str(i)
            else:
                str_i = str(i)
            #   look for the line
            for row in lines:
                target = f'p0{str_i}'
                if row.lstrip().startswith(target):
                    print(f"Matched line: {row}")
                    idx_lst = [int(i) for i in row.split('|')[1].split()]
                    #   append to the list of partition_idxs_lst
                    #   the format: partition, partition_str, row_start, row_end, col_start, col_end
                    partition_idxs_lst.append([i, 'p0' + str_i, idx_lst[-2], idx_lst[-1],
                                               idx_lst[-4], idx_lst[-3]])

                    
print("Partition index list:", partition_idxs_lst)
#   read the first UCN file to get time steps
ucnobj = bf.UcnFile(os.path.join(main_dir, 'MT3D001.UCN.p000'))
time_steps = ucnobj.get_times()
print("TIME STEPS ", time_steps)

#   loop through the time steps
#for ts in time_steps:
#%%
#   only create the output for the last time step
ts = time_steps[-1]
#   make a 3D array with dimension of nlay (n partitions) and nrow and ncol
partition_arr = np.zeros((len(partition_idxs_lst), nlay, nrow, ncol)) * np.nan
partition_head_arr = np.zeros((len(partition_idxs_lst), nlay, nrow, ncol)) * np.nan
#   now loop through the partition indxs list and assign the partition array to the corresponding vertical layer
for k in range(len(partition_idxs_lst)):
    #   read in the UCN file and get the right timestep
    ucnobj = bf.UcnFile(os.path.join(main_dir, 'MT3D001.UCN.' + partition_idxs_lst[k][1]))
    time_steps = ucnobj.get_times()
    conc_arr = ucnobj.get_data(totim=ts).astype(dtype=np.float64)
    conc_arr[conc_arr > 100.] = np.nan
    partition_arr[k, :, partition_idxs_lst[k][2] - 1 : partition_idxs_lst[k][3],
                  partition_idxs_lst[k][4] - 1 : partition_idxs_lst[k][5]] = conc_arr
    nrow_part, ncol_part = conc_arr.shape[1], conc_arr.shape[2]
#%%

    #   read the heads from the list file as well
    #with open(os.path.join(sp_dir, mini_modelname + '.list.p000'), 'r') as listfile:
    # with open(os.path.join(main_dir, model_name + '.list.' + partition_idxs_lst[k][1]), 'r') as listfile:
    #     lines = listfile.readlines()
    #     #   create a head array for the partition and find the corresponding lines
    #     head_arr = np.zeros((nlay, nrow_part, ncol_part)) * np.nan
    #     #   loop through layers
    #     for z in range(nlay):
    #         #   now look for the string in the list file
    #         #lay_ts_str = 'HEAD IN LAYER' + str(z + 1).rjust(4) + ' AT END OF TIME STEP' + str(time_steps.index(ts) + 1).rjust(4)
    #         #print(lay_ts_str)
    #         lay_str_end = '1'
    #         pattern = re.compile(rf'HEAD IN LAYER\s+{z + 1}\s+AT END OF TIME STEP\s+{time_steps.index(ts) + 1}')
    #         for row in lines:
    #             if pattern.search(row):
    #                 print(f"Found match: {row.strip()}")
    #         #   find the starting and end index
    #         # for row in lines:
    #         #     if row.find(lay_ts_str) > 0:
    #                 st_idx = lines.index(row)
    #                 break
    #         for row in lines[st_idx:]:
    #             if row.find(lay_str_end) == 0:
    #                 end_idx = st_idx + lines[st_idx:].index(row)
    #                 break
    #         #   loop through the selected lines and get all the row, col values into the head_arr
    #         #   first, create a list of row indexes that we will then loop through to extract the data
    #         row_idx_lst = []
    #         lines_sel = lines[st_idx : end_idx]
    with open(os.path.join(main_dir, model_name + '.list.' + partition_idxs_lst[k][1]), 'r') as listfile:
        lines = listfile.readlines()
        head_arr = np.full((nlay, nrow_part, ncol_part), np.nan)

    for z in range(nlay):
        pattern = re.compile(
            rf'HEAD IN LAYER\s+{z + 1}\s+AT END OF TIME STEP\s+1'
        )
    
        st_idx = None
        for i, row in enumerate(lines):
            if pattern.search(row):
                print(f"Found match at line {i}: {row.strip()}")
                st_idx = i
                break
    
        if st_idx is None:
            print(f"No match found for Layer {z+1}, Timestep 1")
            continue
    
        # Find the end of the block using a line that contains only '1'
        end_idx = None
        for j, row in enumerate(lines[st_idx:], start=st_idx):
            if row.strip() == '1':
                end_idx = j
                print(f"Found end of block at line {j}: {row.strip()}")
                break
    
        if end_idx is None:
            print(f"Could not find end of block for Layer {z+1}")
            continue
    
        lines_sel = lines[st_idx:end_idx]
        row_idx_lst = []  # or process lines_sel as needed
#%%
        # Example: print selected lines for inspection
        print(f"--- Extracted lines for Layer {z+1} ---")
        for line in lines_sel:
            print(line.strip())
            print("--------------------------------------")
            for y in range(nrow_part):
                for row in lines_sel:
                    if row.find(str(y + 1).rjust(4)) == 0:
                        row_idx_lst.append(lines_sel.index(row))
            #   since it is a regular grid the number of rows in the text file will always be the same
            #   so calculate the row step and then use it to get the actuall head values.. finally
            row_step = row_idx_lst[1] - row_idx_lst[0]
            for row in row_idx_lst:
                lines_row = lines_sel[row : row + row_step]
                heads_col_lst = []
                for line in lines_row:
                    if lines_sel.index(line) == row:
                        #   append the values but skip the row index which is always the first string
                        heads_col_lst.append([float(num) for num in line.split()][1:])
                    else:
                        heads_col_lst.append([float(num) for num in line.split()])
                final_head_lst = [item for sublist in heads_col_lst for item in sublist]
                #   insert the list into the head_arr
                head_arr[z, row_idx_lst.index(row), :] = final_head_lst
    #   add the partition head array into the overall head array
    partition_head_arr[k, :, partition_idxs_lst[k][2] - 1 : partition_idxs_lst[k][3],
                       partition_idxs_lst[k][4] - 1 : partition_idxs_lst[k][5]] = head_arr

#   create a nanmean array from the partition_arr - along the axis = 0 (which is the number of partitions)
final_arr = np.nanmean(partition_arr, axis=0)
final_head_arr = np.nanmean(partition_head_arr, axis=0)

#   create the output files
sconc_dir = os.path.join(main_dir, '_sconc_arr.npy')
strt_dir = os.path.join(main_dir, '_strt_arr.npy')
np.save(sconc_dir, final_arr, allow_pickle=True)
np.save(strt_dir, final_head_arr, allow_pickle=True)
#   make the netcdf files
tot_time_str = str(sp_time_dur[a] * a + time_end)
conc_nc_dir = os.path.join(netcdf_dir, '_conc_time_' + tot_time_str + '_yrs.nc')
head_nc_dir = os.path.join(netcdf_dir, '_head_time_' + tot_time_str + '_yrs.nc')
#   create the netcdf files
x_coord_lst = np.arange(dx / 2, final_arr.shape[2] * dx + dx / 2, dx).tolist()
y_coord_lst = np.arange(dy / 2, final_arr.shape[1] * dy + dy / 2, dy).tolist()
z_coord_lst = np.linspace(0, final_arr.shape[0], final_arr.shape[0]).tolist()
conc_nc = xr.Dataset(data_vars={'salinity': (('z', 'y', 'x'), final_arr)},
                     coords={'x': x_coord_lst,
                             'y': y_coord_lst,
                             'z': z_coord_lst,
                             'time': sp_time_dur[a] * a + time_end})
conc_nc.to_netcdf(conc_nc_dir)
head_nc = xr.Dataset(data_vars={'gw_head': (('z', 'y', 'x'), final_head_arr)},
                     coords={'x': x_coord_lst,
                             'y': y_coord_lst,
                             'z': z_coord_lst,
                             'time': sp_time_dur[a] * a + time_end})
head_nc.to_netcdf(head_nc_dir)

