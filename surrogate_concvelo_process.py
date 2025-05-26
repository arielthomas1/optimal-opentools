#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:43:40 2025
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
work_dir='/home/ariel2/Projects/optimal_mod_runs/'
os.chdir(work_dir)
import numpy as np
import flopy.utils.binaryfile as bf
import xarray as xr
#import imod
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import sys
import pandas as pd
import re
matplotlib.use('agg')
#%% Inspecting the concvelo file 

with open("/home/ariel2/Projects/optimal_mod_runs/sm_7/concvelo.tec", "r") as f:
    for line in f:
        if "period" in line.lower():
            print(line.strip())
            
with open("/home/ariel2/Projects/optimal_mod_runs/sm_7/concvelo.tec", "r") as f:
    text=f.read()
# Count exact word 'root', case-insensitive
count = len(re.findall(r'\bstress\b', text, re.IGNORECASE))
print(f"The word 'stress' appears {count} times.")


#%%

# RSL at 10ka intervals over the past 2 glacial-interglacial cycles.
sp_sealevel=[ -27.3,  -35.9,  -83.2,  -65.3,  -87.3, -108.7, -101.3,  -77.2,
         -3.3,  -47.2,  -33.6,  -48.4,  -33. ,  -68.6,  -82.3,  -76.9,
        -87.3,  -90.6, -115. ,  -49. ,    0. ]
# 1D array of stress period duraction: 10ka
sp_time = 9999*np.ones_like(sp_sealevel)
perlen = (365.25*9999)*np.ones_like(sp_sealevel)
nstp=200
total_sim_time=sum(perlen)
#   get argument values
model_name = 'sm_7'
model_name_sp = 'sm_7'
main_dir = work_dir
sp_dir = os.path.join(work_dir,model_name)
time_start = 0
time_end = total_sim_time
sp_length = sp_time[0]*365.25
ts_len = sp_length/nstp
ts_sum = int(sp_length / ts_len)
sp_name = 'sm_7'
#hk_arr_dir = sys.argv[9]
#top_arr_dir = sys.argv[10]
#bot_arr_dir = sys.argv[11]
#cleanup_output = sys.argv[12]
#cleanup_input = sys.argv[13]
#txt_out = 0
#peterl_file_template_dir = sys.argv[14]
#sea_level = sys.argv[15]

#%%

#   define grid dimensions for row, col (dx and dy) and lay thickness (dz)
dx = 100
dy = 100
dz = 10
#   define the name of the mini stress period
if time_start == 0:
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
mini_sp_dir = sp_dir

#   create the results directory if it doesnt exist
main_res_dir = os.path.join(main_dir, model_name, 'results')
os.makedirs(main_res_dir, exist_ok = True)

#   load the ibound_arr (if it doesnt exist)
ibound_arr_dir = os.path.join(main_dir, model_name, 'ibound_arr.npy')
ibound_arr = np.load(ibound_arr_dir, allow_pickle = True)

#   get the nlay, nrow and ncol values
nlay, nrow, ncol = ibound_arr.shape[0], ibound_arr.shape[1], ibound_arr.shape[2]

#   create the main output folders
netcdf_dir = os.path.join(main_res_dir, '_netcdf')
os.makedirs(netcdf_dir, exist_ok=True)

#%%
#   Extract the salinity concentrations and heads from the partition list and UCN files
#   find the list.p0000 file
partition_idxs_lst = []
str_start = ' p000 :   '
str_end = ' p011 :   '  #   you need to check that this number matches the last p000x file created, depends on number
                        #   of partitions you used in  your parallel model..
if os.path.isfile(os.path.join(mini_sp_dir, model_name + '.list.p000')):
    with open(os.path.join(mini_sp_dir, model_name + '.list.p000'), 'r') as listfile:
        lines = listfile.readlines()
        for i in range(24):
            str_i = str(i).zfill(2)
            for row in lines:
                if row.strip().startswith(f'p0{str_i} :'):
                    parts = row.split('|')
                    if len(parts) > 1:
                        idx_lst = [int(x) for x in parts[1].split()]
                        partition_idxs_lst.append([i, f'p0{str_i}', idx_lst[-4], idx_lst[-3], idx_lst[-2], idx_lst[-1]])
                        
#%%
# #   go through the concvelo.tec files and extract the information to be stored in the final netcdf file

def concvelo_tec_reader(file, tot_ts):
    """
    file = r'g:\_Models\_ELI_model_HoA\_2D_model\concvelo.tec.p000 '
    tot_ts = 10
    :param file:
    :param tot_ts:
    :return:
    """
    #   starting time step will always be 1
    ts_int = 1
    #   open only the top few lines, to find the initial time step and dimensions
    with open(file, 'r') as a:
        top_lines = [next(a) for _ in range(10)]
        #   loop through the lines
        for line_top in top_lines:
            if 'ZONE T=' in line_top:
                line_str = line_top.split(' ')
                line_str = [i for i in line_str if i != '']
                nx = int(line_str[line_str.index('J=') - 1].replace(',', ''))
                ny = int(line_str[line_str.index('J=') + 1].replace(',', ''))
                try:
                    nz = int(line_str[line_str.index('K=') + 1].replace(',', ''))
                except ValueError:
                    nz_str = [i for i in line_str if 'K=' in i][0]
                    nz = int(nz_str.replace('K=', ''))
                break
    #   create empty arrays for each output type, the indexes in the arrays will be ts, lay, row, col
    head_arr_out, conc_arr_out, vx_arr_out, vy_arr_out, vz_arr_out = np.zeros([tot_ts, nz, ny, nx]),\
                                                                     np.zeros([tot_ts, nz, ny, nx]),\
                                                                     np.zeros([tot_ts, nz, ny, nx]),\
                                                                     np.zeros([tot_ts, nz, ny, nx]),\
                                                                     np.zeros([tot_ts, nz, ny, nx])
    #   go through the file and assign values to the right arrays one by one
    conc_velo_txt = open(file, 'r')
    for line_txt in conc_velo_txt:
        if 'TEXT ' in line_txt:
            ts_int += 1
        else:
            line_str = line_txt.split(',')
            if 'ZONE' not in line_txt and 'TEXT' not in line_txt and 'VARIABLES' not in line_txt:
                # 'VARIABLES= "X", "Y", "Z" , "HEAD" , "CONC" , "VX" , "VY" , "VZ"\n'
                head_arr_out[ts_int - 1, int(line_str[2]) - 1, int(line_str[1]) - 1, int(line_str[0]) - 1] = float(line_str[3])
                conc_arr_out[ts_int - 1, int(line_str[2]) - 1, int(line_str[1]) - 1, int(line_str[0]) - 1] = float(line_str[4])
                vx_arr_out[ts_int - 1, int(line_str[2]) - 1, int(line_str[1]) - 1, int(line_str[0]) - 1] = float(line_str[5])
                vy_arr_out[ts_int - 1, int(line_str[2]) - 1, int(line_str[1]) - 1, int(line_str[0]) - 1] = float(line_str[6])
                vz_arr_out[ts_int - 1, int(line_str[2]) - 1, int(line_str[1]) - 1, int(line_str[0]) - 1] = float(line_str[7])
                #line_number = [ts_int, int(line_str[0]), int(line_str[1]), int(line_str[2]), float(line_str[3]), float(line_str[4]),
                #               float(line_str[5]), float(line_str[6]), float(line_str[7])]
                #arrays.append(line_number)
            else:
                print('Getting HEAD/CONC/Vx/Vy/Vz for partition = ' + file.split('.')[-1] + ' and TS = ' + str(ts_int))
                pass
    conc_velo_txt.close()
    #   return the final arrays
    return head_arr_out, conc_arr_out, vx_arr_out, vy_arr_out, vz_arr_out

#%%
#   get the arrays for all the partitioned models 
#
# p000_arrs = concvelo_tec_reader(os.path.join(mini_sp_dir, 'concvelo.tec.p000'), ts_sum)
# p001_arrs = concvelo_tec_reader(os.path.join(mini_sp_dir, 'concvelo.tec.p001'), ts_sum)
# p002_arrs = concvelo_tec_reader(os.path.join(mini_sp_dir, 'concvelo.tec.p002'), ts_sum)
# p003_arrs = concvelo_tec_reader(os.path.join(mini_sp_dir, 'concvelo.tec.p003'), ts_sum)
# p_lst = [p000_arrs, p001_arrs, p002_arrs, p003_arrs]

num_cores = 12
ts_sum = ts_sum

p_lst = []  # This is now a list instead of a dictionary

for i in range(num_cores):
    filename = f'concvelo.tec.p{str(i).zfill(3)}'
    full_path = os.path.join(mini_sp_dir, filename)
    p_lst.append(concvelo_tec_reader(full_path, ts_sum))
    
#   combine the arrays together, using the partition indexes that we got from the list file p000
#   there is overlap between the partitions, usually 2 cells, for those we will get the nan mean value
#   1) make arrays that will have
ts_num = ts_sum
nlay_tot, nrow_tot, ncol_tot = ibound_arr.shape[0], ibound_arr.shape[1], ibound_arr.shape[2]

#%%
#   Function that puts together the partitioned arrays
"""
lst_partition_idxs, nlay_num, nrow_num, ncol_num, lst_p, idx_p, ts_idx = partition_idxs_lst, nlay_tot, nrow_tot, ncol_tot, p_lst, a, ts
"""


def combine_partition_arrs(lst_partition_idxs, nlay_num, nrow_num, ncol_num, lst_p, idx_p, ts_idx):
    #   now loop through the partition indxs list and assign the partition array to the corresponding vertical layer
    partition_arr_out = np.zeros((len(lst_partition_idxs), nlay_num, nrow_num, ncol_num)) * np.nan
    for t in range(len(lst_partition_idxs)):
        p_arr = lst_p[t][idx_p][ts_idx]
        #   find the right indexes
        col_st = lst_partition_idxs[t][4] - 1
        col_end = col_st + p_arr.shape[-1]
        row_st = lst_partition_idxs[t][2] - 1
        row_end = row_st + p_arr.shape[-2]
        #   assign the array to the partitioned array
        partition_arr_out[t, :, row_st : row_end, col_st : col_end] = p_arr[:, :, :]
    #   get nan mean of the array along the partition axis
    out_arr = np.nanmean(partition_arr_out, axis = 0)
    return out_arr


#   loop through the time steps and array types to create output netcdf files
arr_name_lst = ['head', 'conc', 'Vx', 'Vy', 'Vz']

#   read the first UCN file to get time steps
ucnobj = bf.UcnFile(os.path.join(mini_sp_dir, 'MT3D001.UCN.p000'))
time_steps = ucnobj.get_times()
print("TIME STEPS ", time_steps)
#%%
time_steps_yrs = [int(i / 365.25) for i in time_steps]
for a in range(len(p_lst[0])):
    final_arr = np.zeros((len(time_steps), nlay_tot, nrow_tot, ncol_tot)) * np.nan
    #final_arr = np.zeros((ts_num, nlay_tot, nrow_tot, ncol_tot)) * np.nan
    #for ts in range(ts_num):
    for ts in range(len(time_steps)):
        ts_arr = combine_partition_arrs(partition_idxs_lst, nlay_tot, nrow_tot, ncol_tot, p_lst, a, ts)
        final_arr[ts, :, :, :] = ts_arr
    #   save the file as a netcdf define the output file names
    #out_nc_dir = os.path.join(netcdf_dir, '_' + arr_name_lst[a] + '_time_' + str(ts_yrs_lst[ts]) + '_yrs.nc')
    time_save = time_start + int(time_steps[ts] / 365.25)
    out_nc_dir = os.path.join(netcdf_dir, '_' + arr_name_lst[a] + '_' + sp_name + '.nc')
    print(out_nc_dir)
    #   create the netcdf files
    x_coord_lst = np.arange(dx / 2, final_arr.shape[3] * dx + dx / 2, dx).tolist()
    y_coord_lst = np.arange(dy / 2, final_arr.shape[2] * dy + dy / 2, dy).tolist()
    z_coord_lst = np.linspace(1, final_arr.shape[1], final_arr.shape[1]).tolist()
    out_nc = xr.Dataset(data_vars = {arr_name_lst[a] : (('time', 'z', 'y', 'x'), final_arr)},
                        coords = {'time' : time_steps_yrs,
                                  'x' : x_coord_lst,
                                  'y' : y_coord_lst,
                                  'z' : z_coord_lst})
    out_nc.to_netcdf(out_nc_dir)
    del final_arr
a = 0
final_head_arr = np.zeros((1, nlay_tot, nrow_tot, ncol_tot)) * np.nan
#ts = time_steps[-1]
ts_arr = combine_partition_arrs(partition_idxs_lst, nlay_tot, nrow_tot, ncol_tot, p_lst, a, ts)
final_head_arr[0, :, :, :] = ts_arr
a = 1
final_arr = np.zeros((1, nlay_tot, nrow_tot, ncol_tot)) * np.nan
ts_arr = combine_partition_arrs(partition_idxs_lst, nlay_tot, nrow_tot, ncol_tot, p_lst, a, ts)
final_arr[0, :, :, :] = ts_arr
sconc_dir = os.path.join(mini_sp_dir, '_sconc_arr.npy')
strt_dir = os.path.join(mini_sp_dir, '_strt_arr.npy')
np.save(sconc_dir, final_arr, allow_pickle=True)
np.save(strt_dir, final_head_arr, allow_pickle=True)

#%%
#   make plots for each time step
import os
import numpy as np
import flopy.utils.binaryfile as bf
#import imod
import matplotlib
import matplotlib.pyplot as plt
import flopy
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#   load the necessary input arrays and DIS package info
m = flopy.modflow.Modflow.load(os.path.join(sp_dir, model_name + '.nam_swt'))
top = m.dis.gettop()[0, 0]
bot_elev_lst = m.dis.getbotm()[:, 0, 0].tolist()
#top = np.load(top_arr_dir).tolist()
#bot_elev_lst = np.load(bot_arr_dir).tolist()
out_png_dir = os.path.join(main_res_dir, '_plots')
os.makedirs(out_png_dir, exist_ok = True)
#   function to plot the concentration, heads and fluxes
with xr.open_dataset(os.path.join(netcdf_dir, '_conc_' + sp_name + '.nc')) as conc_in:
    arr_conc = conc_in['conc'].values
arr_conc[arr_conc > 1000.] = np.nan
with xr.open_dataset(os.path.join(netcdf_dir, '_head_' + sp_name + '.nc')) as head_in:
    arr_head = head_in['head'].values
arr_head[arr_head == -9999.99] = np.nan
with xr.open_dataset(os.path.join(netcdf_dir, '_Vx_' + sp_name + '.nc')) as vx_in:
    arr_vx = vx_in['Vx'].values
with xr.open_dataset(os.path.join(netcdf_dir, '_Vz_' + sp_name + '.nc')) as vz_in:
    arr_vz = vz_in['Vz'].values
hk_arr = np.load(hk_arr_dir, allow_pickle=True)[:, :, :]
#   make masks for the gela formation and the conduit
gela_mask = np.copy(hk_arr[:, 0, :])
gela_mask[gela_mask != 0.4155] = 0
gela_mask[gela_mask == 0.4155] = 1
mapimg = (gela_mask == 1)
# a vertical line segment is needed, when the pixels next to each other horizontally
#   belong to diffferent groups (one is part of the mask, the other isn't)
# after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
# the same is repeated for horizontal segments
hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])
# if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
#   (2,7) and (2,8), i.e. from (2,8)..(3,8)
# in order to draw a discountinuous line, we add Nones in between segments
l = []
for p in zip(*hor_seg):
    l.append((p[1], p[0]+1))
    l.append((p[1]+1, p[0]+1))
    l.append((np.nan,np.nan))
# and the same for vertical segments
for p in zip(*ver_seg):
    l.append((p[1]+1, p[0]))
    l.append((p[1]+1, p[0]+1))
    l.append((np.nan, np.nan))
# now we transform the list into a numpy array of Nx2 shape
segments = np.array(l)
# now we need to know something about the image which is shown
#   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
#   drawn with origin='lower'
# with this information we can rescale our points
segments[:, 0] = 1 + (gela_mask.shape[-1] - 1) * segments[:, 0] / mapimg.shape[1]
segments[:, 1] = top + (-9790.0 - top) * segments[:, 1] / mapimg.shape[0]
gela_salinity_pct = []
#   loop through the time steps
for ts in range(arr_conc.shape[0]):
    #   get time string for saving the plot for given time step
    time_ts = int(time_steps[ts] / 365.25) + time_start
    time_ts = int(math.ceil(time_ts / 100) * 100.0)
    if time_ts < 1000:
        time_ts_str = '000' + str(time_ts)
    elif 1000 <= time_ts < 10000:
        time_ts_str = '00' + str(time_ts)
    elif 10000 <= time_ts < 100000:
        time_ts_str = '0' + str(time_ts)
    else:
        time_ts_str = str(time_ts)
    print(ts, time_ts_str)
    #   Make CSV files based on the input from Petrel - so I J K X Y Z salinity, for each time step
    #       create the results directory if it doesnt exist
    csv_out_dir = os.path.join(main_res_dir, '_csv')
    os.makedirs(csv_out_dir, exist_ok=True)
    if txt_out == 1:
        #       define directory with the input csv file to use as template
        #   read in the csv file with active geology for the stress period
        fields = ['i', 'j', 'k', 'x', 'y', 'z', 'porosity']
        df_sp_in = pd.read_csv(peterl_file_template_dir, header=7, sep=",", index_col=False, names=fields)
        df_out = df_sp_in.copy()
        #   add column
        df_out['salinity_TDS_g/l'] = -9999.
        df_out['gw_heads_m'] = -9999.
        #   loop through the selected rows and then update the salinity column
        #   in the loop we have to use the name "row_" with underscore because one of the columns in the dataframe is called row..
        for index, row_ in df_out.iterrows():
            lay = int(row_.k - min(df_out['k']))
            row = 0     # int(row_.i) - 1
            col = int(row_.j - min(df_out['j']))
            #   update the row in the main dataframe based on the index
            if not math.isnan(arr_conc[ts, lay, row, col]):
                df_out.at[index, 'salinity_TDS_g/l'] = round(arr_conc[ts, lay, row, col], 2)
                df_out.at[index, 'gw_heads_m'] = round(arr_head[ts, lay, row, col], 2)
        #   delete the unnecessary columns
        df_out = df_out.drop(columns=['porosity'])
        df_out.i = df_out.i.astype(int)
        df_out.j = df_out.j.astype(int)
        df_out.k = df_out.k.astype(int)
        #   define the path for exporting
        path = os.path.join(csv_out_dir, '_ts_' + time_ts_str + '_yrs.gslib')
        # export DataFrame to text file
        with open(path, 'a') as f:
            #   first write the header lines so it can be directly read in the great Petrel
            f.write("PETREL:\n")
            f.write("8\n")
            f.write("i_index unit1 scale1\n")
            f.write("j_index unit1 scale1\n")
            f.write("k_index unit1 scale1\n")
            f.write("x_coord unit1 scale1\n")
            f.write("y_coord unit1 scale1\n")
            f.write("z_coord unit1 scale1\n")
            f.write("sal unit1 scale1\n")
            f.write("hyd_head unit1 scale1\n")
            #   write the Pandas Dataframe into the text file without any header or index..
            df_string = df_out.to_string(header=False, index=False)
            df_string = df_string.strip('\n').split('\n')
            df_string_final = [' '.join(i.split()) for i in df_string]
            f.write('\n'.join(df_string_final))
            f.close()
    #   define the figure
    #   create the Y list of mid-cell positions
    cell_thk = top - bot_elev_lst[0]
    elev_lst = [top] + bot_elev_lst
    y_lst = []
    for i in range(len(elev_lst) - 1):
        y_lst.append(elev_lst[i] - cell_thk / 2)
    ncol, nlay = arr_conc.shape[-1], arr_conc.shape[1]
    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    #   define cmap for the Hk values
    cmaps = [plt.cm.jet, plt.cm.viridis, plt.cm.copper]
    titles = ['Salinity concentration (TDS in mg/l)',
              'Groundwater heads (m)',
              'Geology (Hk in m/d) with flow vectors']
    hk_plot = np.copy(hk_arr[:, 0, :])
    hk_plot[hk_plot == 0] = np.nan
    heads_plot = np.copy(arr_head[ts, :, 0, :])
    heads_plot[heads_plot == -999.99] = np.nan
    arr_lst = [arr_conc[ts, :, 0, :], heads_plot, hk_plot]
    bounds_salinity = [0, 1, 5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 35, 50, 70, 105]
    bounds_heads = np.arange((math.floor(np.nanmin(heads_plot) / 100) * 100), (math.ceil(np.nanmax(heads_plot) / 100) * 100 + 100), 100)
    bounds_hk = [0.0000001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1., 10.]
    #   calculate the number of cells in Gela within salinity intervals
    out_csv_cols = ['sp', 'time']
    tot_gela_cells = 0
    for g in range(len(bounds_salinity) - 1):
        out_csv_cols.append(str(bounds_salinity[g]) + ' to ' + str(bounds_salinity[g + 1]))
    #   now get the
    salinity_arr = arr_conc[ts, :, 0, :]
    salinity_lst = []
    for i in range(salinity_arr.shape[0]):
        for j in range(salinity_arr.shape[0]):
            if gela_mask[i, j] == 1:
                tot_gela_cells += 1
                #print(i, j, salinity_arr[ i, j])
                salinity_lst.append(salinity_arr[i, j])
    #   check max depth of several concentrations - fresh (up to 1mg/l, mild brackish - up to 10mg/l
    #   highly brackish up to 20 mg/l and saline 35 mg/l)
    #   mask the array by the gela aquifer
    salinity_gela_arr = np.copy(salinity_arr)
    #   make the values nan if its not in gela aquifer
    salinity_gela_arr[gela_mask == 0] = np.nan
    #   now for each value find the maximum depth below sea level
    conc_lst, depth_lst = [1., 10., 20., 35.], []
    for conc_val in conc_lst:
        out_csv_cols.append('Depth (m bsl) to ' + str(conc_val) + ' mg/l TDS')
        bot_elev = np.nan
        for i in range(salinity_gela_arr.shape[0] - 1, 0, -1):
            conc_vals = salinity_gela_arr[i, :].tolist()
            if len([h for h in conc_vals if h <= conc_val]) > 0:
                bot_elev = y_lst[i] + dz / 2
                #depth_lst.append(bot_elev)
                break
        depth_lst.append(bot_elev)
    #   now for each interval count the number of cells - and then get pct from total Gela cells
    salinity_pct = []
    for g in range(len(bounds_salinity) - 2):
        count_cells = len([k for k in salinity_lst if bounds_salinity[g] <=  k < bounds_salinity[g + 1]])
        salinity_pct.append(round(100 * count_cells / tot_gela_cells, 1))
    #   the last one
    count_cells_last = len([k for k in salinity_lst if k > bounds_salinity[-2]])
    salinity_pct.append(round(100 * count_cells_last / tot_gela_cells, 1))
    gela_salinity_pct.append([sp_name, time_ts] + salinity_pct + depth_lst)
    #[-10000, -5000, -2500, -1000, -750, -500, -400, -300, -200,
    # -100, -75, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300, 400, 500,
    # 750, 1000, 2500, 5000, 10000]
    bounds_lst = [bounds_salinity, bounds_heads, bounds_hk]
    # [int(h) for h in np.linspace(-999, 999, 20).tolist()]
    for row in range(3):
        ax = axs[row]
        print(row, ts)
        cmap = cmaps[row]
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = bounds_lst[row]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax.axhline(y=sea_level, color='black', linestyle='-', linewidth=0.5)
        im = ax.imshow(arr_lst[row], aspect='auto', interpolation='none',
                       extent = (1, arr_lst[row].shape[-1], y_lst[-1], y_lst[0]), cmap=cmap, norm=norm)
        if row == 2:
            # add the flow vectors
            arr_vx[np.abs(arr_vx) == 0.] = np.nan
            arr_vz[np.abs(arr_vz) == 0.] = np.nan
            qx1_avg = np.empty((arr_vx.shape[1], arr_vx.shape[-1]), dtype=arr_vx.dtype)
            qz1_avg = np.empty((arr_vz.shape[1], arr_vz.shape[-1]), dtype=arr_vz.dtype)
            qx1_avg[:, :] = 0.5 * (arr_vx[ts, :, 0, 0: ncol] + arr_vx[ts, :, 0, : ncol])
            qx1_avg[:, 0] = 0.5 * arr_vx[ts, :, 0, 0]
            qz1_avg[:, :] = 0.5 * (arr_vz[ts, 0 : nlay, 0, :] + arr_vz[ts, : nlay, 0, :])
            qz1_avg[0, :] = 0.5 * arr_vz[ts, 0, 0, :]
            y, x, z = m.dis.get_node_coordinates()
            z = np.linspace(top, y_lst[-1], arr_lst[row].shape[0])
            x = np.linspace(1, arr_lst[row].shape[-1], arr_lst[row].shape[-1])
            X, Z = np.meshgrid(x, z)
            # X, Z = np.meshgrid(x, z[:, 0, 0])
            #   define the distances for each flowline in x and z directions
            x_q_dst = 83
            y_q_dst = 1000
            iskip_x = 3  # int(x_q_dst / 50)
            iskip_z = 3  # int(y_q_dst / 10)
            ax.axhline(y = sea_level, color = 'black', linestyle= '-', linewidth = 0.5)
            im1 = ax.imshow(arr_lst[row], aspect='auto', interpolation='none', cmap=cmap, norm=norm,
                             extent=(1, arr_lst[row].shape[-1], y_lst[-1], y_lst[0]))
            if arr_vx is not None and arr_vz is not None:
                ax.quiver(X[::iskip_z, ::iskip_x], Z[::iskip_z, ::iskip_x], qx1_avg[::iskip_z, ::iskip_x],
                           -qz1_avg[::iskip_z, ::iskip_x],
                           color='grey', scale=None, headwidth=3, headlength=3, headaxislength=2, width=0.0015)
                ax.quiver(X[::iskip_z, ::iskip_x], Z[::iskip_z, ::iskip_x], qx1_avg[::iskip_z, ::iskip_x],
                           -qz1_avg[::iskip_z, ::iskip_x],
                           color='grey', scale=None, headwidth=3, headlength=3, headaxislength=2, width=0.0015)
        ax.plot(segments[:, 0], segments[:, 1], color='k', linewidth=0.75)
        ax.set_title(titles[row])
        if row == 2:
            cb = plt.colorbar(im, ticks = bounds, ax = ax, format = matplotlib.ticker.FormatStrFormatter('%.6f'), aspect = 20)
            cb.ax.tick_params(labelsize=7)
        else:
            cb = plt.colorbar(im, ticks = bounds, ax = ax, aspect = 20)
            cb.ax.tick_params(labelsize=7)
        #cb.ax.set_xticks(bounds)
        #cb.ax.set_xticklabels([str(p) for p in bounds])  # horizontal colorbar
        #cb1 = fig.colorbar(im, ax = ax, format = matplotlib.ticker.FormatStrFormatter('%.5f'))
        #cb1.set_label('%.5f', labelpad=-40, y=1.05, rotation=0)
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.print_figure(os.path.join(out_png_dir, '_ts_' + time_ts_str), dpi = 500)
#   save the final csv file
out_dir = os.path.join(main_res_dir, sp_name + '_gela_slainity_pct.csv')
df_out = pd.DataFrame(gela_salinity_pct, columns = out_csv_cols)
df_out.to_csv(out_dir, index = False)
#   delete files if specified
if cleanup_output:
    for item in os.listdir(sp_dir):
        if item.endswith((".tec", ".list", ".UCN", ".hds", ".cbc")):
            try:
                os.remove(os.path.join(sp_dir, item))
            except (IsADirectoryError, PermissionError):
                pass
if cleanup_input:
    for item in os.listdir(sp_dir):
        if not item.endswith(".npy"):
            try:
                os.remove(os.path.join(sp_dir, item))
            except (IsADirectoryError, PermissionError):
                pass
"""
#   read the Vz, Vy, Vx array and calculate the maximum step size criteria
with xr.open_dataset(os.path.join(netcdf_dir, '_Vz_time_10_yrs.nc')) as vz_in:
    vz_arr = vz_in['Vz'].values
with xr.open_dataset(os.path.join(netcdf_dir, '_Vy_time_10_yrs.nc')) as vy_in:
    vy_arr = vy_in['Vy'].values
with xr.open_dataset(os.path.join(netcdf_dir, '_Vx_time_10_yrs.nc')) as vx_in:
    vx_arr = vx_in['Vx'].values
#   this is the array with cell thickness in m
thk_arr = ibound_arr * dz
#   get the values for the first time step
vz_ts1 = vz_arr[0, :, :, :]
vy_ts1 = vy_arr[0, :, :, :]
vx_ts1 = vx_arr[0, :, :, :]
max_stepsize_arr = 1 / (abs(vx_ts1 / dx) + abs(vy_ts1 / dy) + abs(vz_ts1 / thk_arr))
max_stepsize_arr[max_stepsize_arr == np.inf] = np.nan
print(np.nanmin(max_stepsize_arr), np.nanmax(max_stepsize_arr), np.nanmean(max_stepsize_arr))
stepsize_count = 100
low_stepsize_idx = np.argpartition(max_stepsize_arr.flatten(), stepsize_count)
low_stepsize_vals = np.sort(max_stepsize_arr.flatten()[low_stepsize_idx[:stepsize_count]])
lay_idx, row_idx, col_idx = np.where(max_stepsize_arr == low_stepsize_vals[0])
"""
#   read the first UCN file to get time steps
#ucnobj = bf.UcnFile(os.path.join(mini_sp_dir, 'MT3D001.UCN.p000'))
#time_steps = ucnobj.get_times()
#print("TIME STEPS ", time_steps)
#   loop through the time steps
#for ts in time_steps:
"""
#   only create the output for the last time step
ts = time_steps[-1]
time_save = time_start + int(ts / 365.25)
print('Saving model output for time ' + str(time_save))
#   make a 3D array with dimension of nlay (n partitions) and nrow and ncol
partition_arr = np.zeros((len(partition_idxs_lst), nlay, nrow, ncol)) * np.nan
partition_head_arr = np.zeros((len(partition_idxs_lst), nlay, nrow, ncol)) * np.nan
#   now loop through the partition indxs list and assign the partition array to the corresponding vertical layer
for k in range(len(partition_idxs_lst)):
    #   read in the UCN file and get the right timestep
    ucnobj = bf.UcnFile(os.path.join(mini_sp_dir, 'MT3D001.UCN.' + partition_idxs_lst[k][1]))
    time_steps = ucnobj.get_times()
    conc_arr = ucnobj.get_data(totim=ts).astype(dtype=np.float64)
    conc_arr[conc_arr > 100.] = np.nan
    partition_arr[k, :, partition_idxs_lst[k][2] - 1 : partition_idxs_lst[k][3],
                  partition_idxs_lst[k][4] - 1 : partition_idxs_lst[k][5]] = conc_arr
    nrow_part, ncol_part = conc_arr.shape[1], conc_arr.shape[2]
    #   read the heads from the list file as well
    #with open(os.path.join(mini_sp_dir, mini_modelname + '.list.p000'), 'r') as listfile:
    with open(os.path.join(mini_sp_dir, model_name + '.list.' + partition_idxs_lst[k][1]), 'r') as listfile:
        print(os.path.join(mini_sp_dir, model_name + '.list.' + partition_idxs_lst[k][1]))
        lines = listfile.readlines()
        #   create a head array for the partition and find the corresponding lines
        head_arr = np.zeros((nlay, nrow_part, ncol_part)) * np.nan
        #   loop through layers
        for z in range(nlay):
            #   now look for the string in the list file
            lay_ts_str = 'HEAD IN LAYER' + str(z + 1).rjust(4) + ' AT END OF TIME STEP' + str(time_steps.index(ts) + 1).rjust(4)
            #print(lay_ts_str)
            lay_str_end = '1'
            #   find the starting and end index
            for row in lines:
                if row.find(lay_ts_str) > 0:
                    st_idx = lines.index(row)
                    break
            for row in lines[st_idx:]:
                if row.find(lay_str_end) == 0:
                    end_idx = st_idx + lines[st_idx:].index(row)
                    break
            #   loop through the selected lines and get all the row, col values into the head_arr
            #   first, create a list of row indexes that we will then loop through to extract the data
            row_idx_lst = []
            lines_sel = lines[st_idx : end_idx]
            for y in range(nrow_part):
                for row in lines_sel:
                    if row.find(str(y + 1).rjust(4)) == 0:
                        row_idx_lst.append(lines_sel.index(row))
            #   since it is a regular grid the number of rows in the text file will always be the same
            #   so calculate the row step and then use it to get the actuall head values.. finally
            if len(row_idx_lst) > 1:
                #row_step = row_idx_lst[1] - row_idx_lst[0]
                row_idx_end = -1
            else:
                for row in lines_sel:
                    if ' HEAD WILL BE SAVED ON UNIT   30 AT END OF TIME STEP' in row:
                        row_idx_end = lines_sel.index(row) - 1
                    else:
                        row_idx_end = -1
                #row_step = row_idx_end - row_idx_lst[0]
            for row in row_idx_lst:
                if row_idx_end == -1:
                    lines_row = lines_sel[row :]
                else:
                    lines_row = lines_sel[row:row_idx_end]
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
final_head_arr[final_head_arr == -1000] = 0
#   create the output files
if ts == time_steps[-1]:
    sconc_dir = os.path.join(mini_sp_dir, '_sconc_arr.npy')
    strt_dir = os.path.join(mini_sp_dir, '_strt_arr.npy')
    np.save(sconc_dir, final_arr, allow_pickle=True)
    np.save(strt_dir, final_head_arr, allow_pickle=True)
#   make the netcdf files
tot_time_str = str(time_save)  #str(sp_time_dur[a] * a + time_end)
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
                             'time': time_save})  #sp_time_dur[a] * a + time_end})
conc_nc.to_netcdf(conc_nc_dir)
head_nc = xr.Dataset(data_vars={'gw_head': (('z', 'y', 'x'), final_head_arr)},
                     coords={'x': x_coord_lst,
                             'y': y_coord_lst,
                             'z': z_coord_lst,
                             'time': time_save})  #sp_time_dur[a] * a + time_end})
head_nc.to_netcdf(head_nc_dir)
"""
""" easy plotting.. """
"""
arr_in = final_arr
nanval = -1000
cmap_in = cmap_salinity
bounds_in = bounds
"""
"""
def plot_3d(top_arr_3d, bot_arr_3d, arr_in, nanval, cmap_in):
    arr_plot = np.copy(arr_in)
    arr_plot[arr_plot == nanval] = np.nan
    #   flatten the array and create a list - so we can plot scalar values in color ranges
    scalar_lst = arr_plot.flatten().tolist()
    scalar_lst = [i for i in scalar_lst if not math.isnan(i)]
    #   assign the colors to values
    #   create an xarray from the z_array - has to fit the IMOD standards
    da = xr.DataArray(data = arr_plot,
                      dims = ["layer", "y", "x"],
                      coords = {"top" : (["layer", "y", "x"], top_arr_3d),
                                "bottom" : (["layer", "y", "x"], bot_arr_3d),
                                "layer" : np.arange(1, arr_plot.shape[0] + 1),
                                "y" : np.arange(1, 1 + arr_plot.shape[1]) * dy,
                                "x" : np.arange(1, 1 + arr_plot.shape[2]) * dx})
    #   create the voxels and plot the 3D grid
    z_grid = imod.visualize.grid_3d(da, vertical_exaggeration=50, exterior_only=False, exterior_depth=1, return_index=False)
    z_grid.cell_data['salinity'] = scalar_lst
    z_grid.plot(scalars = 'salinity', show_grid = True, window_size=[1600, 800], cmap=cmap_in)
#   plotting salinity
final_arr[final_arr < 0] = 0
final_arr[final_arr > 35.] = 35.
cmap_salinity = plt.cm.get_cmap("jet", 35)
plot_3d(top_arr_3d, bot_arr_3d, final_arr, -1000, cmap_salinity)
"""
"""
TODO : figure out how to normalize the colormap so it matches the bounds below that are usually used for salinity mapping.
#   define salinity color ranges
cmap_salinity = plt.cm.jet
cmaplist = [cmap_salinity(i) for i in range(cmap_salinity.N)]
cmap_salinity = cmap_salinity.from_list('Custom cmap', cmaplist, cmap_salinity.N)
# define the bins and normalize
bounds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 35.0]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap_salinity.N)