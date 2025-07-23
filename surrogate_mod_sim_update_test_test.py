# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:57:10 2025

@author: Ariel
"""
import os
#work_dir=r"H:\My Drive\OPTIMAL\Project work\optimal"
work_dir='/home/ariel2/Projects/optimal'
os.chdir(work_dir)
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import geone
import geone.covModel as gcm
import geone.imgplot3d as imgplt3
import pyvista as pv
# pv.set_jupyter_backend('static')
import sys
import subprocess
import yaml
import optimal_functions as of
import ArchPy as ap
ap_path=ap.__file__
sys.path.append(ap_path)
from ArchPy.base import * #ArchPy main functions
from ArchPy.tpgs import * #Truncated plurigaussians
from ArchPy.inputs import * #Truncated plurigaussians
import ArchPy.ap_mf
from ArchPy.ap_mf import archpy2modflow, array2cellids
import flopy as fp
from scipy.interpolate import PchipInterpolator


# #Definig model data folder i.e. where ArchPy surrogate models are stroed
# mod_fol=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections\ArchPy_mods" # text files containing all the model parameters
# mod_data=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections\surrogate_mod_summary" # text files containing all the model parameters
# output_data=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections" # main folder with all output data including figures
#modflow_ws="H:/My Drive/OPTIMAL/Project work/optimal/surrogate_sections/surrogate_simulations"
# modflow_ws=r"C:\Users\Ariel\sciebo\OPTIMAL_LOCAL"
# imod_path=r"C:\Users\Ariel\sciebo\imod_files\iMODexe\iMOD-WQ_V5_3_SVN359_X64R.exe"
# imod6_path=r"C:\Users\Ariel\sciebo\imod_files\iMODexe\MODFLOW6_v6.2.1.exe"
# seawat_exe=r"C:\Users\Ariel\sciebo\SEAWAT\swt_v4_00_05\exe\swt_v4x64.exe"
# mpich_exe=r"C:\Program Files (x86)\MPICH2\bin\mpiexec.exe"

#Linux setup

mod_fol='/home/ariel2/Projects/optimal/surrogate_sections/ArchPy_mods'
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
output_data='/home/ariel2/Projects/optimal/surrogate_sections'

modflow_ws='/data/optimal/mod_files' # Formerly /home/ariel2/Projects/optimal_mod_runs'
imod_path='/home/ariel2/software/bin/seawat_svn387'
#imod6_path=
seawat_exe='/home/ariel2/software/swtv4'
mpich_exe='/opt/intel/oneapi/mpi/2021.11/bin/mpiexec.exe'
fig_fol='/home/ariel2/Projects/optimal/surrogate_sections/figures'

#%%

#user-defined number of models to be retrieved
num_models=5
mod_dict= {}
for i in range(1, num_models + 1):
    mod_id =  f'sm_{i}'  # Generate model ID
    mod_loc=os.path.join(mod_fol, "ap_{}".format(mod_id))
    mod_dict[mod_id]=import_project(mod_id,mod_loc)


#%%Load and verify model
# i=2
# mod_loc=os.path.join(mod_fol, "ap_{}".format(mod[i]))
# mod_obj=import_project(mod[i],mod_loc)

for i in range(1,num_models+1):
    SEED = np.random.randint(239)
    rng = np.random.default_rng(SEED)
    mod_id = f'sm_{i}'  # Generate model ID

    # Extracting porosity and hydraulic conductivity
    por_facies1=mod_dict[mod_id].get_prop('Por')[0,0,0,:,0,:]
    comp_facies1=of.apply_porosity_compaction(mod_dict, mod_id, -0.0005)
    strati=mod_dict[mod_id].get_units_domains_realizations()[0,:,0,:]
    flow_par=mod_dict[mod_id].get_prop('K')[0,0,0,:,0,:]
    # Define a shared color scale ignoring NaN values
    vmin = 0.1
    vmax = 0.65
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    
    # Plot Porosity Field
    im0 = axes[0,0].imshow(np.flipud(strati), cmap="viridis", aspect="auto", origin="upper")
    axes[0,0].set_title("Stratigraphy")
    
    # Plot Hyd Conductivity
    im1 = axes[0,1].imshow(np.flipud(flow_par), cmap="hot", aspect="auto", origin="upper")
    axes[0,1].set_title("Hydraulic Conductivity")
    
    
    axes[1,0].set_ylabel("layer")
    im2 = axes[1,0].imshow(np.flipud(por_facies1), cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axes[1,0].set_title("Original Porosity Field")
    
    axes[1,0].set_ylabel("layer")
    
    # Plot Compacted Porosity Field
    im3 = axes[1,1].imshow(np.flipud(comp_facies1[:,0,:]), cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axes[1,1].set_title("Compacted Porosity Field")
    axes[1,1].set_xlabel("Model column")
    
    # Add a shared colorbar
    cbar = fig.colorbar(im2, ax=axes[1,1], orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Porosity (-)")
    
    cbar = fig.colorbar(im1, ax=axes[0,1], orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Hydraulic Conductivity (m/day)")
    
    fig.suptitle(f"Surrogate model: {mod_id}", fontsize=18)
    
    #Saving Figure
    #fig.savefig('{}/figures/{}_summary.png'.format(output_data,mod_id), dpi=450, bbox_inches='tight')

    # # Reading sealevel data and plotting 
    
    # #df_sl=pd.read_csv("H:/My Drive/OPTIMAL/Project work/Global_datasets/Sealevel_data_Imbrie_200k.csv")
    # df_sl=pd.read_csv('/home/ariel2/global_datasets/Sealevel_data_Imbrie_200k.csv')
    # h = -15
    # lvl=h*np.ones([len(df_sl['Age [ka]'].values),1])
    
    # fig,ax=plt.subplots(figsize=(9,4))
    
    # plt.plot(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,color='black', label='Relative Sea level')
    # plt.plot(df_sl['Age [ka]'].values,lvl,'b--',linewidth=1, label='model elevation')
    # plt.fill_between(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,h, where=df_sl['RSL [m]'].values <= h,label='Recharge',facecolor='palegreen', interpolate=True)
    # plt.fill_between(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,h, where=df_sl['RSL [m]'].values > h,label='Flooding',facecolor='lightcoral', interpolate=True)
    
    # #plt.step(df_sl['Age'].values,df_sl['Sealevel'].values,where='mid',linewidth=0.5)
    # plt.xlabel('Age (ka BP)',fontsize='16')
    # plt.title('Chengsi model relative sea level',fontsize='16')
    # plt.ylabel('Relative sea level (m)',fontsize='16')
    # ax.set_xlim(0,200)
    # plt.legend()
    # plt.show()


    # #Preparing sealevel data NB. Comment out, this only needed to be done once
    
    # # Reading dating from Imbrie et al.
    # df_sl=pd.read_csv("/home/ariel2/global_datasets/Sealevel_data_Imbrie_200k.csv")
    # #convertig to np array format
    # sl_arr=df_sl.to_numpy()
    # #resampling data at 10,000 year intervals 
    # sl_10k_sp = sl_arr[::5, :] # 10000 year intervals
    # sl_4k_sp = sl_arr[::2,:] # 4000 year intervals
    # # formating sea level array for time stepping
    # sp_sealevels=sl_10k_sp[:,2][::-1].round(1)
    # sp_sealevels_4k=sl_4k_sp[:,2][::-1].round(1)


    #% Create the basic MODFLOW model structure

#MOdflow model grid setup
   
    #retrieving model name from dict
    sm=mod_dict[mod_id]
    
    nlay, nrow, ncol = sm.get_nz(), sm.get_ny(), sm.get_nx()
    delr, delc = sm.get_sx(), sm.get_sy()
    z_vals=sm.get_zg()
    dz = sm.get_sz()
    top_elev=top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
    #botm = np.arange(top_elev, np.nanmin(z_vals) - dz, -dz)#[1:]   
    botm_full =(np.flipud(z_vals)-dz)[:-1]  # flip the array to have the same orientation as the ArchPy table and drops the last index since archpy uses cell centers
    #Creating ibound array with all cells set to zero. 
    #cells will be activated in the model properties block
    ibound_sm=np.zeros((nlay,nrow,ncol))
    #Truncating the ibound array at 1km depth for more efficient simulation time. 
    ibound_sm.shape

  
 

    # Model properties
    factor=1
    # Retrieving the Hydraulic conductivity array from surrogate model
    hk_arr=factor*mod_dict[mod_id].get_prop('K')[0,0,0,:,:,:]
    # Retriving the porosity array from the surrogate model and applying porosity compaction
    por_arr=of.apply_porosity_compaction(mod_dict, mod_id, -0.0005)
    
    print("Shape of por_arr:", por_arr.shape)
    print("Shape of ibound_sm before update:", ibound_sm.shape)
    # Update ibound: set to 1 where porosity is NOT NaN
    hk_arr=np.flipud(hk_arr)
    por_arr=np.flipud(por_arr)
    
    #Defining active cells by masking the porosity array
    ibound_sm = np.where(~np.isnan(por_arr), 1, ibound_sm)
    #Find the index of the last active column
    toe_cut_off=of.find_last_col(ibound_sm,1)
    #Calculating the topography of the model to determine the top array for dis package
    top_arr=of.find_first_active(ibound_sm[:,:,0:toe_cut_off]) #finding the index of the first active cell in each col
    top_elev_arr=top_elev-(top_arr*dz) #calculating the top elevation relative to top of the model
    
    
    cut_off_idx = np.where(botm_full == -1000)[0][0]
    botm = botm_full[:cut_off_idx + 1].copy()    
    print("Shape of bot elev array:", botm.shape)
    nlay=cut_off_idx+1
    ncol=toe_cut_off
    #Creating ibound array with all cells set to zero. 
    #cells will be activated in the model properties block
    #deactivating model cells below 1km
    ibound_sm=ibound_sm[0:nlay,:,0:ncol]
    hk_arr=hk_arr[0:nlay,:,0:ncol]
    por_arr=por_arr[0:nlay,:,0:ncol]
    botm=botm[0:nlay]
    print("Shape of ibound_sm after update:", ibound_sm.shape)
    print("Shape of por array  after update:", por_arr.shape)
    print("Shape of hk array  after update:", hk_arr.shape)
    
    
    # Reshape the porosity array to 2D
    porosity_2d = por_arr.squeeze()
    
    # Determine the extent of your model (assuming unit spacing)
    extent = [0, porosity_2d.shape[1], porosity_2d.shape[0], 0]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8,3))
    
    # Plot the porosity data using imshow
    im = ax.imshow(porosity_2d, cmap='viridis', extent=extent, aspect='auto') # Using 'viridis' colormap
    
    # Add labels and title
    ax.set_xlabel('layers')
    ax.set_ylabel('columns')
    ax.set_title(f'Porosity Distribution - {mod_id}')
    
    # Add a colorbar to show porosity values
    cbar = fig.colorbar(im, ax=ax, label='Porosity')
    
    # Show the plot
    plt.show()
   # fig.savefig(f'{fig_fol}/porosity_{mod_id}.jpg',dpi=450, bbox_inches='tight')
    # Reshape the porosity array to 2D
    hk_2d = hk_arr.squeeze()
    
    # Determine the extent of your model (assuming unit spacing)
    extent = [0, hk_2d.shape[1], hk_2d.shape[0], 0]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6,3))
    
    # Plot the porosity data using imshow
    im = ax.imshow(hk_2d, cmap='viridis', extent=extent, aspect='auto') # Using 'viridis' colormap
    
    # Add labels and title
    ax.set_xlabel('X-direction (Column)')
    ax.set_ylabel('Z-direction (Row)')
    ax.set_title('Hydraulic Head')
    
    # Add a colorbar to show porosity values
    cbar = fig.colorbar(im, ax=ax, label='Hydraulic Head')
    
    # Show the plot
    plt.show()
    
    # iBOUND QC
    # Assuming your ibound array is named 'ibound' and has shape [294, 1, 1600]
    #ibound_sm[:, :, 0][ibound_sm[:, :, 0] == 1] = -1
    ibound = ibound_sm
   
    # Reshape the array to 2D by removing the middle dimension of size 1
    ibound_2d = ibound.squeeze()[0:nlay,:]
    
    # Create a masked array to handle different cell types for plotting
    masked_active = np.ma.masked_where(ibound_2d != 1, ibound_2d)
    masked_inactive = np.ma.masked_where(ibound_2d != 0, ibound_2d)
    masked_negative = np.ma.masked_where(ibound_2d !=-1, ibound_2d)
    
    # Determine the extent of your model (assuming unit spacing for now)
    extent = [0, ibound_2d.shape[1], 8*ibound_2d.shape[0], 0] # [xmin, xmax, ymax, ymin] for imshow
    

    # Create the plot
    fig2, ax2 = plt.subplots()
    
    # Plot the different cell types with distinct colors
    im_active = ax2.imshow(masked_active, cmap=plt.cm.viridis, vmin=0.5, vmax=1.5, extent=extent, label='Active (1)')
    im_inactive = ax2.imshow(masked_inactive, cmap=plt.cm.gray, vmin=-0.5, vmax=0.5, extent=extent, label='Inactive (0)')
    im_negative = ax2.imshow(masked_negative, cmap=plt.cm.autumn, vmin=-1.5, vmax=-0.5, extent=extent, label='Constant (-)')
    
    # Add labels and titlen 
    ax2.set_xlabel('columns')
    ax2.set_ylabel('layers')
    ax2.set_title('iBound Array Visualization (z vs x)')
    
    # Create custom patches for the legend
    active_patch = mpatches.Patch(color=plt.cm.viridis(0.75), label='Active (1)')
    inactive_patch = mpatches.Patch(color=plt.cm.gray(0.5), label='Inactive (0)')
    negative_patch = mpatches.Patch(color=plt.cm.autumn(0.75), label='Constant (-1)')
    
    # Add the legend
    #ax2.legend(handles=[active_patch, inactive_patch, negative_patch])
    
    
    # Add the legend
    ax2.legend(handles=[active_patch, inactive_patch, negative_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.show()
    #fig.savefig('{}/figures/{}_ibound.png'.format(output_data,mod_id), dpi=450, bbox_inches='tight')

    #% Initial conditions and stress periods
    sp_len=3999
    # RSL at 10ka intervals over the past 2 glacial-interglacial cycles.
    # sp_sealevel=[ -27.3,  -35.9,  -83.2,  -65.3,  -87.3, -108.7, -101.3,  -77.2,
    #          -3.3,  -47.2,  -33.6,  -48.4,  -33. ,  -68.6,  -82.3,  -76.9,
    #         -87.3,  -90.6, -115. ,  -49. ,    0. ]
    sp_sealevel=[ -27.3,  -14. ,  -23.2,  -49.6,  -80.5,  -83.2,  -70.1,  -65.6,
            -66.5,  -74. ,  -87.3, -100.7, -109. , -104. ,  -99.5, -101.3,
           -102.2,  -93.3,  -54.4,   -5.9,   -3.3,  -16.6,  -40.4,  -46.9,
            -41.6,  -33.6,  -38.3,  -46.6,  -48.1,  -47.8,  -33. ,  -41.6,
            -55. ,  -81.7,  -91.8,  -82.3,  -74.3,  -73.4,  -78.7,  -82. ,
            -87.3,  -90.6,  -90. ,  -92.4, -103.1, -115. , -109. ,  -70.7,
            -31.8,  -10.4,    0. ]
    # 1D array of stress period duraction: 10ka
    sp_time = sp_len*np.ones_like(sp_sealevel)
    
    # model name and folder structure
    model_dir = os.path.join(modflow_ws, mod_id)
    #creating directories
    os.makedirs(model_dir, exist_ok = True)
    #   create the SEAWAT model object and start creating individual packages
    swt = fp.seawat.Seawat(mod_id,'nam_swt', exe_name=seawat_exe, model_ws=model_dir)
    # TODO Check imod_path


    #Starting concentrations and hyd. heads
    init_sl_idx= np.where(botm == 0)[0][0] #returns the index of the initial index of the sealevel i.e., the postion of the present day sealevel in the onshore-offshore transect. This should occur where botm array is zero
    
    #starting concentration 
    sconc_arr = ibound_sm*35
    #initializing model with all cells above sealevel set to freshwater saturation
    sconc_arr[0:init_sl_idx,:,:]*=0
    #starting head
    head_arr = ibound_sm*0
    

   
    #************************
    #% DIS package
    #************************
    perlen = (365.25*sp_len)*np.ones_like(sp_sealevel) #an array filled with values of the length of each stress period in days.
    nstp = 400*np.ones_like(sp_sealevel) # number of time steps per period ca. 500 years per time step (???)
    nper = len(sp_sealevel) # number of stress periods
    
    dis = fp.modflow.ModflowDis(swt, nlay, nrow, ncol, nper = nper, delr = delr, delc = delc, 
                                top = top_elev_arr, botm = botm, perlen = perlen, nstp = nstp)

    #************************
    #% BAS Package
    #************************
    bas = fp.modflow.ModflowBas(swt, ibound = ibound_sm, strt = head_arr)

    #************************
    #% LPF Package
    #************************
    # setting anisotropy to the hyd. conductivity field
    vk_arr = 0.1*hk_arr 
    lpf = fp.modflow.ModflowLpf(swt, laytyp = 0, hk = hk_arr, vka = vk_arr, ipakcb = 1)

    #************************

    # BOUNDARY CONDITIONS
    #   create the icbund array
    icbund_sm = ibound_sm
    
    rch_val = 0.001
    rch_arr = np.zeros((ibound_sm.shape[1],ibound_sm.shape[2]))
    # Initialize empty dictionaries to store rch,drn,ghb, chd and ssm arrays per stress period
    rch_data = {}     
    drn_data = {} 
    ghb_arr_in = {}
    ssm_arr_in = {}
    chd_arr_in = {}
    #
    itype = fp.mt3d.Mt3dSsm.itype_dict()
    #getting the smooth top DEM for assigning drainage level
    tos=of.read_mod_file(mod_data,mod_id, 'Toe of slope')
    inland_sect=of.read_mod_file(mod_data,mod_id, 'Inland length') #Extracting model inland length from summary file
    sbd=of.read_mod_file(mod_data,mod_id, 'Shelf break depth') #Extracting model shelf break depth from summary file
    sw=of.read_mod_file(mod_data,mod_id, 'Shelf width') #Extracting model shelf width from summary file
    mod_len=of.read_mod_file(mod_data, mod_id, 'Total model length')
    
    x_mod = np.array([0, int(inland_sect), int(inland_sect + sw), int(mod_len)])
    z_top = np.array([top_elev, 0, -sbd, -tos])
    spline_top = PchipInterpolator(x_mod, z_top)
    x_new = np.linspace(x_mod.min(), x_mod.max(), ncol)
    top_elev_dem = spline_top(x_new)
    
    # # Create the plot
    # fig, ax = plt.subplots(figsize=(6,3))
    # plt.plot(top_elev_arr[0,:])
    # plt.plot(top_elev_dem)
    # plt.show()
    # Assuming your ibound array is named 'ibound' and has shape [294, 1, 1600]
    ibound_temp=ibound_sm.copy()
    ibound_temp[:, :, 0][ibound_temp[:, :, 0] == 1] = -1   
 
    cond_damper=1 #factor to dampen the conductivity of boundary cells in case of numerical instability
    for a in range(len(sp_sealevel)):
    
        sea_level=sp_sealevel[a]
        
        ghb_input_lst = []
        chd_input_lst = []
        ssmdata = []
        inland_head=0
        #recharge and drainage
        rch_arr_sp = np.zeros((ibound_sm.shape[1],ibound_sm.shape[2]))
        drn_input_lst = []
        
        # the inland part on the edges of the active model domain will be assigned the topographical head
        # for each column check the first active cell - and if it is above sea level then assign fresh head
        row=0
        #lay_idx = [0, nlay - 1]
        for i in range(ncol):
            #   select the active cells only
                col_cells = [lay for lay in range(nlay) if ibound_sm[lay, row, i] == 1.0]
                if len(col_cells) > 0: 
                    lay_idx = ibound_sm[:, row, i].tolist().index(1) #returns the first index in the column where ibound is 1                                 
                    actual_top_of_highest_active_cell = botm[lay_idx]+dz     
                    if  actual_top_of_highest_active_cell > sea_level: #TODO double check this if artifacts sdtill persist
                        cond_val = cond_damper*vk_arr[lay_idx, row, i]*1000 #i.e (delc * delr) / dz 
                        #ghb_input_lst.append([lay_idx, row, i, rel_elev, cond_val])
                        #ssmdata.append([lay_idx, row, i, 0.0, itype['RCH']])
                        #recharge
                        rch_arr_sp[row, i] = rch_val
                        #drainag
                        drn_input_lst.append([lay_idx, row, i, top_elev_dem[i], cond_val]) 
                        print(f'actual top - {actual_top_of_highest_active_cell}, top elev - {top_elev_arr[row,i]}')
                    if actual_top_of_highest_active_cell <= sea_level:
                        cond_val = cond_damper*vk_arr[lay_idx, row, i]*1000 # #i.e (delc * delr) / dz
                        ghb_input_lst.append([lay_idx, row, i, sea_level, cond_val]) #the GHB head is what the boundary water level is, i.e., 0.0 m.
                        ssmdata.append([lay_idx, row, i, 35.0, itype['GHB']])

        #Adding chb to inland boundary cells
        #rel_top_elev=dz*sea_level_idx #calculating the elevation of the top elevationcell relative to the new sealevel.
        for k in range(nlay):
            if ibound_temp[k, 0, 0] == -1.0:
                cond_val = hk_arr[k, 0, 0] * 20 # (2*dz*delr)/delc
                ghb_input_lst.append([k, 0, 0, top_elev, cond_val])
                ssmdata.append([k, row, 0, 0.0, itype['GHB']])
        # Directly assign to the dictionaries for the current stress period 'a'
        rch_data[a]= rch_arr_sp.copy()  
        drn_data[a] = drn_input_lst
        ghb_arr_in[a] = ghb_input_lst
        ssm_arr_in[a] = ssmdata 
        chd_arr_in[a] = chd_input_lst
                   
    #************************
    # GHB package
    #************************

    ghb = fp.modflow.ModflowGhb(swt, ipakcb = 1, stress_period_data = ghb_arr_in)

    #************************

    # RCH & DRN package

    drn = fp.modflow.ModflowDrn(swt, ipakcb=1, stress_period_data=drn_data)

    rch = fp.modflow.ModflowRch(swt, nrchop = 3, ipakcb = 1, rech = rch_data)
    
    # #************************
    # # CHD package
    # #************************
    # chd = fp.modflow.ModflowChd(swt, stress_period_data=chd_arr_in) # Pass your prepared data here

    
    # #************************

    # OUTPUT CONTROL
    
    #************************

    #   write the OC package
    ihedfm = 1  # a code for the format in which heads will be printed.
    iddnfm = 0  # a code for the format in which drawdowns will be printed.
    extension = ['oc', 'hds', 'ddn', 'cbc']
    unitnumber = [14, 30, 52, 51]
    #   create the dictionary that defines how to write the output file
    spd = {(0, 0): ['SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET', 'SAVE HEADTEC', 'SAVE CONCTEC',
                    'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC']}
    for t in range(nper):  # Iterate through each stress period
        per = t  # Stress period number (t starts from 0, corresponding to the 1st period)
        
        # Define the last timestep for the current stress period (200th timestep, which is index 199)
        last_step = nstp[t]-1  # last timestep in the stress period (indexing starts at 0)
        
        # Specify output control for the final timestep of each stress period
        spd[(per, last_step)] = [
            'SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET',
            'SAVE HEADTEC', 'SAVE CONCTEC', 'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC'
        ]
    oc = fp.modflow.ModflowOc(swt, ihedfm=ihedfm, stress_period_data=spd, unitnumber=unitnumber, compact=True)

    #************************

    # BTN Package

    #************************

    #   the BTN package
    porosity = por_arr
    dt0 = 0
    nprs = 1
    ifmtcn = 0
    chkmas = False
    nprmas = 10
    nprobs = 10
    total_sim_time=sum(perlen)
    #timprs_lst = list(np.linspace(1,total_sim_time,nper,endpoint=True,dtype=int))
    timprs_lst = np.cumsum(perlen).tolist()
    btn = fp.mt3d.Mt3dBtn(swt, nprs=nprs, timprs=timprs_lst, prsity=porosity, sconc=sconc_arr,
                             ifmtcn=ifmtcn, chkmas=chkmas, nprobs=nprobs, nprmas=nprmas, dt0=dt0)
    
    #************************

    #ADV Package
    #************************
    # write the ADV package
    adv = fp.mt3d.Mt3dAdv(swt, mixelm=0, mxpart=2000000)
    
    #************************

    # DSP Package
    #************************

    #   write the DSP package
    dmcoef = 0.0000864 # effective molecular diffusion coefficient [M2/D]
    al = 1. # Longitudinal dispersivity
    trpt = 0.1
    trpv = 0.1
    dsp = fp.mt3d.Mt3dDsp(swt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)
    
    #************************

    #%VDF Package
    #************************

    #   write the VDF package
    iwtable = 0
    densemin = 1000.
    densemax = 1025.
    denseref = 1000.
    denseslp = 0.7143
    firstdt = 0.001
    vdf = fp.seawat.SeawatVdf(swt, iwtable=iwtable, densemin=densemin, densemax=densemax,
                                 denseref=denseref, denseslp=denseslp, firstdt=firstdt)

    #************************

    # SSM Package
    #************************

    #   write the SSM package
    ssm_rch_in = np.copy(rch_arr_sp)*0 
    ssm_rch_all={i: ssm_rch_in.copy() for i in range(nper)}
    ssm = fp.mt3d.Mt3dSsm(swt, crch=0, stress_period_data=ssm_arr_in)

    #************************
   
    # Write simulation
    #************************

    #   write packages and run model
    swt.write_input()

    #************************

    #% Writing files
    #************************
    
    #   write the ascii file with vertical sum of active cells in IBOUND
    ibound_arr_sum = np.sum(ibound_sm, axis=0, dtype=np.int32)
    ibound_arr_sum = ibound_arr_sum.astype(str)
    with open(os.path.join(model_dir, 'LOAD.ASC'), 'wb') as f:
        f.write(ibound_arr_sum)
    
    #   create the pksf and pkst files - change it in case the grid discretization changes
    pksf_lines = ['ISOLVER 1', 'NPC 2', 'MXITER 500', 'RELAX .98', 'HCLOSEPKS 0.0001', 'RCLOSEPKS 1000', 'PARTOPT 0',
                  'PARTDATA', 'external 40 1. (free) -1', 'GNCOL {}'.format(ncol), 'GNROW {}'.format(nrow), 
                  'GDELR', '{}'.format(delr), 'GDELC', '{}'.format(delc),'NOVLAPADV 2', 'END']
    pkst_lines = ['ISOLVER 2', 'NPC 2', 'MXITER 2000', 'INNERIT 50', 'RELAX .98', 'RCLOSEPKS 1.0E-02',
                  'HCLOSEPKS 0.001', 'RELATIVE-L2NORM', 'END']
    # 'CCLOSEPKS=0.00001'
    
    with open(os.path.join(model_dir, mod_id + '.pksf'), 'w') as f:
        for line in pksf_lines:
            f.write(line)
            f.write('\n')
    
    with open(os.path.join(model_dir, mod_id + '.pkst'), 'w') as f:
        for line in pkst_lines:
            f.write(line)
            f.write('\n')
    
    #   open the nam_swt file and append these three lines
    nam_lines = ['PKSF          	  27 ' + mod_id + '.pksf', 'PKST              35 ' +
                 mod_id + '.pkst', 'DATA 40 LOAD.ASC']
    
    with open(os.path.join(model_dir, mod_id + '.nam_swt'), 'a') as f:
        for line in nam_lines:
            f.write(line)
            f.write('\n')
    
    #   save the ibound_arr (if it doesnt exist)
    ibound_arr_dir = os.path.join(model_dir, 'ibound_arr.npy')
    if not os.path.isfile(ibound_arr_dir):
        np.save(ibound_arr_dir, ibound_sm)
        
    print(f'Model {mod_id} files written successfully')
#%% Creat Batch file

#with open(os.path.join(model_dir,'opt_runmod_par.bat'))
#########LINUX#######################    
    # Writing Linux shell script
    #with open('runmod_parallel.sh','w') as infile: 
     #   infile.write("#! /bin/sh  \n")
      #  infile.write("cd Models/StrPer_{} \n".format(model_nr))
       # infile.write("mpirun -np 1 --output-filename runfile /home/ariel/software/bin/seawat_svn387 chengsi_mod_{}.run \n".format(model_nr))
        
    #subprocess.run(["./runmod.sh"],capture_output=True)
    #print('Stress period {} completed'.format(model_nr))
#mod_file=os.path.join(model_dir, mod_id + '.nam_swt')

#####WINDOWS##########################
# #    #Writing the windows batch script
# with open('runmod_parallel.bat','w') as infile:
#     infile.write("\"{}\" -localonly 4 \"{}\" \"{}\"".format(mpich_exe,imod_path,mod_file))
# infile.close()    

#subprocess.call([r'runmod_parallel.bat'])
