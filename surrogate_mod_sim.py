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

modflow_ws='/home/ariel2/Projects/optimal_mod_runs'
imod_path='/home/ariel2/software/bin/seawat_svn387'
#imod6_path=
seawat_exe='/home/ariel2/software/swtv4'
mpich_exe='/opt/intel/oneapi/mpi/2021.11/bin/mpiexec.exe'

#%%

#user-defined number of models to be retrieved
num_models=10
mod_dict= {}
for i in range(1, num_models + 1):
    mod_id =  f'sm_{i}'  # Generate model ID
    mod_loc=os.path.join(mod_fol, "ap_{}".format(mod_id))
    mod_dict[mod_id]=import_project(mod_id,mod_loc)


#%%Load and verify model
# i=2
# mod_loc=os.path.join(mod_fol, "ap_{}".format(mod[i]))
# mod_obj=import_project(mod[i],mod_loc)

mod_id='sm_7'

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
fig.savefig('{}/figures/{}_summary.png'.format(output_data,mod_id), dpi=450, bbox_inches='tight')

#%%

#df_sl=pd.read_csv("H:/My Drive/OPTIMAL/Project work/Global_datasets/Sealevel_data_Imbrie_200k.csv")
df_sl=pd.read_csv('/home/ariel2/global_datasets/Sealevel_data_Imbrie_200k.csv')
h = -15
lvl=h*np.ones([len(df_sl['Age [ka]'].values),1])

fig,ax=plt.subplots(figsize=(9,4))

plt.plot(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,color='black', label='Relative Sea level')
plt.plot(df_sl['Age [ka]'].values,lvl,'b--',linewidth=1, label='model elevation')
plt.fill_between(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,h, where=df_sl['RSL [m]'].values <= h,label='Recharge',facecolor='palegreen', interpolate=True)
plt.fill_between(df_sl['Age [ka]'].values,df_sl['RSL [m]'].values,h, where=df_sl['RSL [m]'].values > h,label='Flooding',facecolor='lightcoral', interpolate=True)

#plt.step(df_sl['Age'].values,df_sl['Sealevel'].values,where='mid',linewidth=0.5)
plt.xlabel('Age (ka BP)',fontsize='16')
plt.title('Chengsi model relative sea level',fontsize='16')
plt.ylabel('Relative sea level (m)',fontsize='16')
ax.set_xlim(0,200)
plt.legend()
plt.show()


#%% Preparing sealevel data NB. Comment out, this only needed to be done once

# # Reading dating from Imbrie et al.
# df_sl=pd.read_csv("H:/My Drive/OPTIMAL/Project work/Global_datasets/Sealevel_data_Imbrie_200k.csv")
# #convertig to np array format
# sl_arr=df_sl.to_numpy()
# #resampling data at 10,000 year intervals 
# sl_10k_sp = sl_arr[::5, :]
# # formating sea level array for time stepping
# sp_sealevels=sl_10k_sp[:,2][::-1].round(1)


#%% Create the basic MODFLOW model structure

#MOdflow model grid setup

#retrieving model name from dict
sm=mod_dict[mod_id]

nlay, nrow, ncol = sm.get_nz(), sm.get_ny(), sm.get_nx()
delr, delc = sm.get_sx(), sm.get_sy()
z_vals=sm.get_zgc()
dz = sm.get_sz()
top_elev=top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
botm = np.arange(top_elev, np.nanmin(z_vals) - dz, -dz)[1:]  
botm = np.flip(np.flipud(botm), axis=0)  # flip the array to have the same orientation as the ArchPy table
print("Shape of bot elev array:", botm.shape)
#Creating ibound array with all cells set to zero. 
#cells will be activated in the model properties block
ibound_sm=np.zeros((nlay,nrow,ncol))
ibound_sm.shape

#%% Model properties

# Retrieving the Hydraulic conductivity array from surrogate model
hk_arr=mod_dict[mod_id].get_prop('K')[0,0,0,:,:,:]
# Retriving the porosity array from the surrogate model and applying porosity compaction
por_arr=of.apply_porosity_compaction(mod_dict, mod_id, -0.0005)

print("Shape of por_arr:", por_arr.shape)
print("Shape of ibound_sm before update:", ibound_sm.shape)
# Update ibound: set to 1 where porosity is NOT NaN
ibound_sm = np.where(~np.isnan(por_arr), 1, ibound_sm)
print("Shape of ibound_sm after update:", ibound_sm.shape)


#%% Initial conditions and stress periods

# RSL at 10ka intervals over the past 2 glacial-interglacial cycles.
sp_sealevel=[ -27.3,  -35.9,  -83.2,  -65.3,  -87.3, -108.7, -101.3,  -77.2,
         -3.3,  -47.2,  -33.6,  -48.4,  -33. ,  -68.6,  -82.3,  -76.9,
        -87.3,  -90.6, -115. ,  -49. ,    0. ]
# 1D array of stress period duraction: 10ka
sp_time = 9999*np.ones_like(sp_sealevel)

# model name and folder structure
model_dir = os.path.join(modflow_ws, mod_id)
#creating directories
os.makedirs(model_dir, exist_ok = True)
#   create the SEAWAT model object and start creating individual packages
swt = fp.seawat.Seawat(mod_id,'nam_swt', exe_name=seawat_exe, model_ws=model_dir)
# TODO Check imod_path
#%% Starting concentrations and hyd. heads

#starting concentration 
sconc_arr = ibound_sm*35
#starting head
head_arr = ibound_sm*0
#%% DIS package

perlen = (365.25*9999)*np.ones_like(sp_sealevel) #an array filled with values of the length of each stress period in days.
nstp = 200*np.ones_like(sp_sealevel) # number of time steps per period ca. 500 years per time step (???)
nper = len(sp_sealevel) # number of stress periods

dis = fp.modflow.ModflowDis(swt, nlay, nrow, ncol, nper = nper, delr = delr, delc = delc, top = top_elev,
                               botm = botm, perlen = perlen, nstp = nstp)

#%% BAS Package

bas = fp.modflow.ModflowBas(swt, ibound = ibound_sm, strt = head_arr)

#%% LPF Package

# setting anisotropy to the hyd. conductivity field
vk_arr = hk_arr * 0.1 
lpf = fp.modflow.ModflowLpf(swt, laytyp = 0, hk = hk_arr, vka = vk_arr, ipakcb = 1)

#%% BOUNDARY CONDITIONS
#   create the icbund array
icbund_sm = ibound_sm
#TODO Create a global list of lists
ghb_input_all = []
chb_input_all = []
ssmdata_all = []
#
itype = fp.mt3d.Mt3dSsm.itype_dict()
#   create the GHB package input here, also start creating the SSM package
for a in range(len(sp_sealevel)):
    sea_level=sp_sealevel[a]
    ghb_input_lst = []
    chb_input_lst = []
    ssmdata = []
    #initiating the first stress period
    sp=0 

    # the inland part on the edges of the active model domain will be assigned the topographical head
    # for each column check the first active cell - and if it is above sea level then assign fresh head
    
    row_idx = [0, nrow - 1]
    for i in range(ncol):
        #   select the active cells only
        for row in row_idx:
            col_cells = [t for t in ibound_sm[:, row, i].tolist() if t == 1]
            if len(col_cells) > 0:
                lay_idx = ibound_sm[:, row, i].tolist().index(1)
                if ibound_sm[lay_idx, row, i] == 1 and botm[lay_idx] + dz >= sea_level:
                    for k in range(nlay):
                        if botm[k] + dz >= sea_level:
                            cond_val = hk_arr[k, row_idx, i] * dz * 1000
                            ghb_input_lst.append([k, row_idx, i, botm[k] + dz, cond_val])
                            ssmdata.append([k, row_idx, i, 0.0, itype['GHB']])
                            icbund_sm[k, row_idx, i] = -1
    # now check for the offshore domain and set all the cells below sea level to saltwater concentration and
    # head equal to sea level. Only in the top layer
    for i in range(nrow):
        for j in range(ncol):
            lay_cells = [t for t in ibound_sm[:, i, j].tolist() if t == 1]
            if len(lay_cells) > 0:
                lay_idx = ibound_sm[:, i, j].tolist().index(1)
                if ibound_sm[lay_idx, i, j] == 1 and botm[lay_idx] + dz < sea_level:
                    cond_val = (vk_arr[lay_idx, i, j] * delc * delr) / dz
                    ghb_input_lst.append([lay_idx, i, j, sea_level, cond_val])
                    ssmdata.append([lay_idx, i, j, 35.0, itype['GHB']])
    ghb_input_all.append(ghb_input_lst)
    ssmdata_all.append(ssmdata)            
                    
#   write the final output dictionary, inlcude each stress period
ghb_arr_in = {}
for d in range(len(perlen)):
    ghb_arr_in[d] = ghb_input_all[d]
# creating a global list to store the ssm input arrays for each stress period    
ssm_arr_in = {}
for e in range(len(perlen)):
    ssm_arr_in[e] = ssmdata_all[e]
    
ghb = fp.modflow.ModflowGhb(swt, ipakcb = 1, stress_period_data = ghb_arr_in)
#%% RCH & DRN package

'''shape of rch array should be (nper,ncol)'''
rch_val = 0.00025
rch_arr = np.zeros((ibound_sm.shape[1],ibound_sm.shape[2]))
drn_input_lst = []
rch_input_lst = []
#   only apply recharge to the cells above sea level, for each column find the first active layer
for i in range(ibound_sm.shape[1]):
    for j in range(ibound_sm.shape[2]):
        lay_lst = [t for t in ibound_sm[:, i, j].tolist() if t == 1]
        if len(lay_lst) > 0:
            top_act_lay = ibound_sm[:, i, j].tolist().index(1)
            top_act_elev = top_elev - top_act_lay * dz
            #   if the top elevation of the cell is above sea level then assign recharge to it
            if top_act_elev >= sea_level:
                #print(top_act_elev)
                rch_arr[i, j] = rch_val
                #drainage
                cond_cell = (vk_arr[top_act_lay, i, j] * delc * delr) / dz
                drn_input_lst.append([int(top_act_lay), i, j, top_act_elev, cond_cell])
              
# creating a list of rch arrays equal to the length of the no of stress period.
rch_arr_in = {}
for c in range(len(perlen)):
    rch_arr_in[c] = rch_arr
rch = fp.modflow.ModflowRch(swt, nrchop = 3, ipakcb = 1, rech = rch_arr_in)
#   write the final output dictionary, inlcude each stress period
if len(drn_input_lst) > 0:
    drn_arr_in = {}
    for c in range(len(perlen)):
        drn_arr_in[c] = drn_input_lst
    drn = fp.modflow.ModflowDrn(swt, ipakcb=1, stress_period_data=drn_arr_in)


#%% OUTPUT CONTROL
#   write the OC package
ihedfm = 1  # a code for the format in which heads will be printed.
iddnfm = 0  # a code for the format in which drawdowns will be printed.
extension = ['oc', 'hds', 'ddn', 'cbc']
unitnumber = [14, 30, 52, 51]
#   create the dictionary that defines how to write the output file
spd = {(0, 0): ['SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET', 'SAVE HEADTEC', 'SAVE CONCTEC',
                'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC']}
for t in range(1, nper):
    per = t  # + 1
    #   to save space on disk, every 10th timestep is saved
    spd[(per, nstp[t])] = ['SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET', 'SAVE HEADTEC', 'SAVE CONCTEC',
                          'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC']
oc = fp.modflow.ModflowOc(swt, ihedfm=ihedfm, stress_period_data=spd, unitnumber=unitnumber, compact=True)

#%% BTN Package

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
                         ifmtcn=ifmtcn, chkmas=chkmas, nprobs=nprobs, nprmas=nprmas, dt0=0)
#%% ADV Package
#   write the ADV package
adv = fp.mt3d.Mt3dAdv(swt, mixelm=0, mxpart=2000000)

#%% DSP Package
#   write the DSP package
dmcoef = 0.0000864  # effective molecular diffusion coefficient [M2/D]
al = 1.
trpt = 0.1
trpv = 0.1
dsp = fp.mt3d.Mt3dDsp(swt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)
#%%VDF Package
#   write the VDF package
iwtable = 0
densemin = 1000.
densemax = 1025.
denseref = 1000.
denseslp = 0.7143
firstdt = 0.001
vdf = fp.seawat.SeawatVdf(swt, iwtable=iwtable, densemin=densemin, densemax=densemax,
                             denseref=denseref, denseslp=denseslp, firstdt=firstdt)
#%% SSM Package
#   write the SSM package
ssm_rch_in = np.copy(rch_arr) * 0.0
ssm_rch_all=np.broadcast_to(ssm_rch_in,(nper,1800))
ssm = fp.mt3d.Mt3dSsm(swt, crch=ssm_rch_in, stress_period_data=ssm_arr_in)

#%% Write simulation
#   write packages and run model
swt.write_input()
#%% Writing files
#   write the ascii file with vertical sum of active cells in IBOUND
ibound_arr_sum = np.sum(ibound_sm, axis=0, dtype=np.int32)
ibound_arr_sum = ibound_arr_sum.astype(str)
with open(os.path.join(model_dir, 'LOAD.ASC'), 'wb') as f:
    f.write(ibound_arr_sum)

#   create the pksf and pkst files - change it in case the grid discretization changes
pksf_lines = ['ISOLVER 1', 'NPC 2', 'MXITER 200', 'RELAX .98', 'HCLOSEPKS 0.001', 'RCLOSEPKS 10000.0', 'PARTOPT 0',
              'PARTDATA', 'external 40 1. (free) -1', 'GNCOL {}'.format(ncol), 'GNROW {}'.format(nrow), 
              'GDELR', '{}'.format(delr), 'GDELC', '{}'.format(delc),'NOVLAPADV 2', 'END']
pkst_lines = ['ISOLVER 2', 'NPC 2', 'MXITER 1000', 'INNERIT 50', 'RELAX .98', 'RCLOSEPKS 1.0E-04',
              'HCLOSEPKS 1.0E+12', 'RELATIVE-L2NORM', 'END']
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
mod_file=os.path.join(model_dir, mod_id + '.nam_swt')

#####WINDOWS##########################
# #    #Writing the windows batch script
# with open('runmod_parallel.bat','w') as infile:
#     infile.write("\"{}\" -localonly 4 \"{}\" \"{}\"".format(mpich_exe,imod_path,mod_file))
# infile.close()    

#subprocess.call([r'runmod_parallel.bat'])
