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
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt

import ArchPy as ap
mod_fol='/home/ariel2/Projects/optimal/surrogate_sections/ArchPy_mods'
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
 #%%

os.makedirs("frames", exist_ok=True)
#retriev model data
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
mod_id='sm_1'
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
    ax1.scatter(100*df_ssm_chd['k'].values,-10*df_ssm_chd['i'].values,marker='o',facecolors=None, edgecolors='b', s=2)
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


#%%
mod_id='sm_1'
# Replace with the actual path to your .cbc file
cbc_file = f'/data/optimal/mod_files/{mod_id}/{mod_id}.cbc'

# Create a CellBudgetFile object
try:
    cbc = bf.CellBudgetFile(cbc_file)

    # Get a list of available records (flow terms) and times
    records = cbc.get_unique_record_names()
    print(f"Available records in CBC file: {records}")

    times = cbc.get_times()
    print(f"Available times in CBC file: {times}")

    # Example: Get 'CONSTANT HEAD' or 'DRAINS' or 'STORAGE' flow for the last time
    last_time = times[-1]

    # Note: Record names are case-sensitive and might have trailing spaces
    # You can print records to see exact names: print(records)

    # Example: Get flow to Constant Head boundaries
    ch_flow_data = None
    try:
        ch_flow_data = cbc.get_data(text='CONSTANT HEAD', totim=last_time)[0] # [0] because get_data returns a list of arrays
        print(f"\nConstant Head flow data shape for time {last_time}: {ch_flow_data.shape}")
        # ch_flow_data will often be a numpy array of (ncells_with_ch, 1) or (ncells_with_ch, nlay)
        # You'll likely need to map this back to grid cells for visualization
    except KeyError:
        print("'CONSTANT HEAD' record not found for this time or file.")


    # Example: Get flow to Drains
    drain_flow_data = None
    try:
        drain_flow_data = cbc.get_data(text='DRAINS', totim=last_time)[0]
        print(f"\nDrain flow data shape for time {last_time}: {drain_flow_data.shape}")
    except KeyError:
        print("'DRAINS' record not found for this time or file.")

    # Example: Get volumetric flow between cells (e.g., flow along rows or columns)
    # These are often named 'FLOW RIGHT FACE' and 'FLOW FRONT FACE' for 3D
    # For 2D, it might be 'FLOW RIGHT FACE' (horizontal) and 'FLOW LOWER FACE' (vertical)
    frf_data = None
    try:
        frf_data = cbc.get_data(text='FLOW RIGHT FACE', totim=last_time)[0]
        print(f"\nFlow Right Face data shape for time {last_time}: {frf_data.shape}")
        # This will be (nlay, nrow, ncol-1) or similar, representing flow between columns
        # You can plot this as vectors or a quiver plot if you also get flow front face
    except KeyError:
        print("'FLOW RIGHT FACE' record not found for this time or file.")

    ff_data = None # For vertical flow if your 2D model is a cross-section
    try:
        ff_data = cbc.get_data(text='FLOW LOWER FACE', totim=last_time)[0] # Or 'FLOW FRONT FACE' depending on orientation
        print(f"\nFlow Lower Face data shape for time {last_time}: {ff_data.shape}")
    except KeyError:
        print("'FLOW LOWER FACE' record not found for this time or file.")


    # To map flow data back to grid:
    # FloPy can help with this, especially when loading the entire model
    # using flopy.mf6.modflow.MFModel or flopy.modflow.Modflow

except FileNotFoundError:
    print(f"Error: CBC file not found at {cbc_file}")
except Exception as e:
    print(f"An error occurred: {e}")
    
#%% Cell thickness
num_models=5
mod_dict= {}
for i in range(1, num_models + 1):
    mod_id =  f'sm_{i}'  # Generate model ID
    mod_loc=os.path.join(mod_fol, "ap_{}".format(mod_id))
    mod_dict[mod_id]=import_project(mod_id,mod_loc)

mod_id='sm_5'
sm=mod_dict[mod_id]

nlay, nrow, ncol = sm.get_nz(), sm.get_ny(), sm.get_nx()
delr, delc = sm.get_sx(), sm.get_sy()
z_vals=sm.get_zg()
dz = sm.get_sz()
top_elev=top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
#botm = np.arange(top_elev, np.nanmin(z_vals) - dz, -dz)#[1:]   
botm =(np.flipud(z_vals)-dz)[:-1]  # flip the array to have the same orientation as the ArchPy table and drops the last index since archpy uses cell centers
print("Shape of bot elev array:", botm.shape)
#Creating ibound array with all cells set to zero. 
#cells will be activated in the model properties block
ibound=np.zeros((nlay,nrow,ncol))
#Truncating the ibound array at 1km depth for more efficient simulation time. 
ibound.shape




# Model properties
factor=1
# Retrieving the Hydraulic conductivity array from surrogate model
hk_arr=factor*mod_dict[mod_id].get_prop('K')[0,0,0,:,:,:]
# Retriving the porosity array from the surrogate model and applying porosity compaction
por_arr=of.apply_porosity_compaction(mod_dict, mod_id, -0.0005)

print("Shape of por_arr:", por_arr.shape)
print("Shape of ibound_sm before update:", ibound.shape)
# Update ibound: set to 1 where porosity is NOT NaN
hk_arr=np.flipud(hk_arr)
por_arr=np.flipud(por_arr)

#Defining active cells by masking the porosity array
ibound = np.where(~np.isnan(por_arr), 1, ibound)
    #deactivating model cells below 1km

#Calculating the topography of the model to determine the top array for dis package
top_arr=of.find_first_active(ibound) #finding the index of the first active cell in each col
top_elev_arr=top_elev-(top_arr*dz) #calculating the top elevation relative to top of the model



# --- 3. Calculate Cell Thicknesses for Active Cells ---
cell_thickness_map = np.full((nlay, ncol), np.nan) # Initialize with NaN for inactive cells

for k in range(nlay):
    for j in range(ncol):
        # For a 2D cross-section, nrow is 1, so row index 'i' is always 0
        i = 0

        if ibound[k, i, j] == 1: # Only calculate for active cells
            # Determine the top elevation of the current cell
            if k == 0:
                # The top of the first layer cell is defined by top_elev_arr
                current_cell_top = top_elev_arr[i, j]
            else:
                # The top of subsequent layers is the bottom of the layer above it
                current_cell_top = botm[k-1]

            # The bottom elevation of the current cell is from the botm array
            current_cell_bottom = botm[k]

            # Calculate thickness
            thickness = current_cell_top - current_cell_bottom

            # Store the thickness in our map
            cell_thickness_map[k, j] = thickness
        else:
            # For inactive cells, keep as NaN or set to 0, depending on visualization preference
            # NaN is good as it won't be colored by imshow
            cell_thickness_map[k, j] = np.nan

# --- 4. Visualize the Cell Thicknesses ---

plt.figure(figsize=(12, 8))

# Use imshow to visualize the cell_thickness_map
# cmap='viridis' is a good default, or 'YlGnBu' for thickness
# origin='upper' is used to make the plot appear with the surface at the top
im = plt.imshow(cell_thickness_map, cmap='viridis', origin='upper', aspect='auto')

# Add color bar
cbar = plt.colorbar(im, label='Cell Thickness (m)')

# Add grid lines to show cell boundaries
plt.grid(True, which='both', color='white', linestyle='-', linewidth=0.5)
# Plot the top surface of the model (continental shelf)
plt.plot(np.arange(ncol), -top_elev_arr[0, :], 'r-', linewidth=2, label='Model Top Surface')

# Set labels and title
plt.xlabel('Column Index')
plt.ylabel('Layer Index (Vertical)')
plt.title('2D Cross-Section: Cell Thickness (Active Cells)')

# A better way to represent elevation on the Y-axis:
# Create custom y-ticks that represent elevation
# We can use the bottom of each layer for the y-tick position
# When origin='upper', layer 0 is at the top (y=0), so its bottom is botm[0].
# As y-index increases downwards, layer index k increases, and botm[k] decreases.
y_tick_positions = np.arange(nlay) # Positions for the top of each layer in imshow coordinates
y_tick_labels = [f"{botm[k]:.0f}" for k in range(nlay)] # Bottom elevation of each layer
# To make the labels correspond to the *bottom* of the layer at that imshow row:
# For origin='upper', row k corresponds to layer k. The bottom of layer k is botm[k].
# So, the y-tick label at y=k should represent botm[k].
plt.yticks(y_tick_positions, y_tick_labels)
plt.ylabel('Bottom Elevation of Layer (m)')

# Adjust y-limits to focus on the active part of the model
# Find min/max active layer indices
active_layers = np.where(~np.isnan(cell_thickness_map))[0]
if len(active_layers) > 0:
    min_active_layer = active_layers.min()
    max_active_layer = active_layers.max()
    #plt.ylim(min_active_layer + 0.5, max_active_layer - 0.5) # Add padding

plt.tight_layout()
plt.show()

# --- 5. Analyze the results ---
print("\n--- Cell Thickness Analysis ---")
active_thicknesses = cell_thickness_map[~np.isnan(cell_thickness_map)]
if active_thicknesses.size > 0:
    print(f"Minimum active cell thickness: {np.min(active_thicknesses):.2f} m")
    print(f"Maximum active cell thickness: {np.max(active_thicknesses):.2f} m")
    print(f"Average active cell thickness: {np.mean(active_thicknesses):.2f} m")
    print(f"Standard deviation of active cell thickness: {np.std(active_thicknesses):.2f} m")
else:
    print("No active cells found to analyze thickness.")

# Check for zero or negative thicknesses (should not happen in a valid grid)
if np.any(active_thicknesses <= 0):
    print("\nWARNING: Found cells with zero or negative thickness! This indicates a grid setup error.")



#%% HEADS

mod_id='sm_1'
# Replace with the actual path to your .hds file
hds_file = f'/data/optimal/mod_files/{mod_id}/{mod_id}.hds'

# Create a HeadFile object
try:
    hds = bf.HeadFile(hds_file)

    # Get a list of available stress periods and times
    times = hds.get_times()
    print(f"Available times in HDS file: {times}")

    # You can get heads for a specific time (e.g., the last time)
    # Or iterate through all times
    last_time = times[-1]
    heads = hds.get_data(totim=last_time)

    print(f"\nHeads data shape for time {last_time}: {heads.shape}")
    # The shape will typically be (nlay, nrow, ncol) for 3D models, or (nrow, ncol) for 2D cross-sections.

    # Example: Plot heads for the first layer (if 3D) or the entire 2D cross-section
    # Assuming your 2D model is (nlay=1, nrow, ncol) or (nrow, ncol)
    if heads.ndim == 3: # For a 3D model, plot a specific layer
        layer_to_plot = 3 # First layer
        plt.figure(figsize=(10, 6))
        plt.imshow(heads[layer_to_plot], cmap='viridis', origin='lower')
        plt.colorbar(label='Head (m)')
        plt.title(f'Heads for Layer {layer_to_plot + 1} at Time {last_time}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()
    elif heads.ndim == 2: # For a 2D cross-section (which your description suggests)
        plt.figure(figsize=(10, 6))
        plt.imshow(heads, cmap='viridis', origin='lower')
        plt.colorbar(label='Head (m)')
        plt.title(f'Heads at Time {last_time}')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

    # You can also get a specific record by layer, row, column
    # For example, head at cell (layer 0, row 0, col 0) at last_time
    # cell_head = heads[0, 0, 0] # For a 3D model
    # print(f"Head at cell (0,0,0): {cell_head}")

except FileNotFoundError:
    print(f"Error: HDS file not found at {hds_file}")
except Exception as e:
    print(f"An error occurred: {e}")
    
for k in range(ch_flow_data.shape[0]):
    cell_head = heads[k, 0, 0] # For a 3D model
    print(f"Head at cell ({k},0,0): {cell_head}")