
import os
import numpy as np
import optimal_functions as of
import matplotlib.pyplot as plt
work_dir='/data/optimal/mod_files'
os.chdir(work_dir)
from matplotlib import colors
import re
import pandas as pd
import imageio.v2 as imageio

#%%
df_mod_runs,completed,failed=of.check_model_runs(work_dir,5)

#%% 

mod_id='sm_1'
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
mod_dir= os.path.join(work_dir,mod_id)
#retrieve model data
top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
# SPlit stress periods
# Set path to your .tec file
file_path =os.path.join(work_dir,mod_id,'concvelo.tec')

# Directory to save the split files
output_dir = os.path.join(work_dir,mod_id)


        
        
# Clean files
of.split_sp_outputs(file_path,output_dir)

num_per=of.count_nper_infile(file_path,search_string='ZONE T=')
for i in range(1, num_per + 1):
    filename = os.path.join(mod_dir, f"results_sp{i}.tec")
    of.clean_tec_file2(filename)
    
#%%
#of.print_last_lines(os.path.join(mod_dir, f"results_sp10_cleaned.tec"))
#%% Visualize
ibound_arr_dir = os.path.join(mod_dir, 'ibound_arr.npy')
ibound_arr = np.load(ibound_arr_dir, allow_pickle = True)

#   get the nlay, nrow and ncol values
nlay, nrow, ncol = ibound_arr.shape[0], ibound_arr.shape[1], ibound_arr.shape[2]

for sp in range(47,48):

    sp_sealevel=[ -27.3,  -14. ,  -23.2,  -49.6,  -80.5,  -83.2,  -70.1,  -65.6,
        -66.5,  -74. ,  -87.3, -100.7, -109. , -104. ,  -99.5, -101.3,
       -102.2,  -93.3,  -54.4,   -5.9,   -3.3,  -16.6,  -40.4,  -46.9,
        -41.6,  -33.6,  -38.3,  -46.6,  -48.1,  -47.8,  -33. ,  -41.6,
        -55. ,  -81.7,  -91.8,  -82.3,  -74.3,  -73.4,  -78.7,  -82. ,
        -87.3,  -90.6,  -90. ,  -92.4, -103.1, -115. , -109. ,  -70.7,
        -31.8,  -10.4,    0. ]

    sl=sp_sealevel[sp-1] #get the corresponding sealevel from the list

    var_names=['X','Y','Z','HEAD','CONC','VX','VY','VZ']
    results=os.path.join(mod_dir, f"results_sp{sp}_cleaned.tec")
    df = pd.read_csv(results,header=None)
    df.columns=var_names
    print(df.head())


    # Extract the X, Z, CONC, and HEAD columns
    x = df['X'].values
    z = df['Z'].values #convert to metersq
    conc = df['CONC'].values
    head = df['HEAD'].values
    valid_head = head[head != -999.99]

    # Get the minimum of the valid values
    min_valid_head = valid_head.min()


    # Determine the dimensions of the grid (assuming it's a regular grid)
    nx = ncol#len(np.unique(x))-1
    nz = nlay#len(np.unique(z))-1

    # Reshape the CONC and HEAD arrays
    conc_2d = conc.reshape((nz, nx))  # Reshape into a 2D array (Z x X)
    head_2d = head.reshape((nz, nx))
    conc_2d =np.where(conc_2d>36,np.nan,conc_2d)
    head_2d =np.where(head_2d<-999,np.nan,head_2d)

    # extracting model geometric parameters
    top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
    base_elev=of.read_mod_file(mod_data,mod_id,'Toe of slope')
    z_vals=np.linspace(top_elev,-1000,5)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharex=True) # Adjust figsize as needed

    # --- Plot 1: Salinity Profile (conc_2d) ---
    im1 = ax1.imshow(conc_2d[0:100,:], extent=[x.min(),x.max(), -1000, 10*z.min()],
                     aspect='auto', vmin=0, vmax=36,cmap='jet')
    ax1.plot(np.arange(1,nx),(sl-top_elev)*np.ones(nx-1),'b--',linewidth=1,label='Sealevel') # adding sealevel position to plot
    fig.colorbar(im1, ax=ax1, label='Salinity (g/L)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_yticks(z_vals)
    ax1.set_yticklabels([f"{val:.1f}" for val in z_vals])
    ax1.set_title(f'Salinity profile - {mod_id} | sp {sp}')
    ax1.legend() # Display the legend for sealevel
    
    # --- Plot 2: HEAD Profile (head_2d) ---
    im2 = ax2.imshow(head_2d[0:100,:], extent=[x.min(), x.max(), -1000, 10*z.min()],
                     aspect='auto',vmin=min_valid_head, vmax=head.max(), cmap='RdBu_r')
    ax2.plot(np.arange(1,nx),(sl-top_elev)*np.ones(nx-1),'b--',linewidth=1,label='Sealevel') # adding sealevel position to plot
    fig.colorbar(im2, ax=ax2, label='Hyd. head (m)')
    ax2.set_xlabel('Distance') # Only the bottom subplot needs an x-label
    ax2.set_ylabel('Elevation (m)')
    ax2.set_yticks(z_vals)
    ax2.set_yticklabels([f"{val:.1f}" for val in z_vals])
    ax2.set_title(f'Hyd. head profile - {mod_id} | sp {sp}')
    ax2.legend() # Display the legend for sealevel
    
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()
    
    # Save or show the plot
    # plt.savefig(f"{os.path.splitext(filename)[0]}_combined_profiles.png")
    plt.show()

#%% Create Video GIF

mod_id='sm_5'
#retriev model data
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
os.chdir(f'/data/optimal/mod_files/{mod_id}')
os.makedirs("gif_results", exist_ok=True)


for sp in range(2,num_per):

    sp_sealevel=[ -27.3,  -14. ,  -23.2,  -49.6,  -80.5,  -83.2,  -70.1,  -65.6,
        -66.5,  -74. ,  -87.3, -100.7, -109. , -104. ,  -99.5, -101.3,
       -102.2,  -93.3,  -54.4,   -5.9,   -3.3,  -16.6,  -40.4,  -46.9,
        -41.6,  -33.6,  -38.3,  -46.6,  -48.1,  -47.8,  -33. ,  -41.6,
        -55. ,  -81.7,  -91.8,  -82.3,  -74.3,  -73.4,  -78.7,  -82. ,
        -87.3,  -90.6,  -90. ,  -92.4, -103.1, -115. , -109. ,  -70.7,
        -31.8,  -10.4,    0. ]

    sl=sp_sealevel[sp-1] #get the corresponding sealevel from the list

    var_names=['X','Y','Z','HEAD','CONC','VX','VY','VZ']
    results=os.path.join(mod_dir, f"results_sp{sp}_cleaned.tec")
    df = pd.read_csv(results,header=None)
    df.columns=var_names
    print(df.head())


    # Extract the X, Z, CONC, and HEAD columns
    x = df['X'].values
    z = df['Z'].values #convert to metersq
    conc = df['CONC'].values
    head = df['HEAD'].values
    valid_head = head[head != -999.99]

    # Get the minimum of the valid values
    min_valid_head = valid_head.min()


    # Determine the dimensions of the grid (assuming it's a regular grid)
    nx = ncol#len(np.unique(x))-1
    nz = nlay#len(np.unique(z))-1

    # Reshape the CONC and HEAD arrays
    conc_2d = conc.reshape((nz, nx))  # Reshape into a 2D array (Z x X)
    head_2d = head.reshape((nz, nx))
    conc_2d =np.where(conc_2d>36,np.nan,conc_2d)
    head_2d =np.where(head_2d<-999,np.nan,head_2d)

    # extracting model geometric parameters
    top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
    base_elev=of.read_mod_file(mod_data,mod_id,'Toe of slope')
    z_vals=np.linspace(top_elev,-1000,5)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6), sharex=True) # Adjust figsize as needed

    # --- Plot 1: Salinity Profile (conc_2d) ---
    im1 = ax1.imshow(conc_2d[0:100,:], extent=[x.min(),x.max(), -1000, 10*z.min()],
                     aspect='auto', vmin=0, vmax=36,cmap='jet')
    ax1.plot(np.arange(1,nx),(sl-top_elev)*np.ones(nx-1),'b--',linewidth=1,label='Sealevel') # adding sealevel position to plot
    fig.colorbar(im1, ax=ax1, label='Salinity (g/L)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_yticks(z_vals)
    ax1.set_yticklabels([f"{val:.1f}" for val in z_vals])
    ax1.set_title(f'Salinity profile - {mod_id} | sp {sp}')
    ax1.legend() # Display the legend for sealevel
    
    # --- Plot 2: HEAD Profile (head_2d) ---
    im2 = ax2.imshow(head_2d[0:100,:], extent=[x.min(), x.max(), -1000, 10*z.min()],
                     aspect='auto',vmin=min_valid_head, vmax=head.max(), cmap='RdBu_r')
    ax2.plot(np.arange(1,nx),(sl-top_elev)*np.ones(nx-1),'b--',linewidth=1,label='Sealevel') # adding sealevel position to plot
    fig.colorbar(im2, ax=ax2, label='Hyd. head (m)')
    ax2.set_xlabel('Distance') # Only the bottom subplot needs an x-label
    ax2.set_ylabel('Elevation (m)')
    ax2.set_yticks(z_vals)
    ax2.set_yticklabels([f"{val:.1f}" for val in z_vals])
    ax2.set_title(f'Hyd. head profile - {mod_id} | sp {sp}')
    ax2.legend() # Display the legend for sealevel
    
    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()
    # Save or show the plot
    # plt.savefig(f"{os.path.splitext(filename)[0]}_combined_profiles.png")
    fig.savefig(f"gif_results/frame_{sp:03d}.png", dpi=150)
    plt.close(fig)
# Now create the GIF
with imageio.get_writer(f'{mod_id}_sim.gif', mode='I', duration=0.5) as writer:
    for sp in range(2,len(sp_sealevel)):
        filename = f"gif_results/frame_{sp:03d}.png"
        image = imageio.imread(filename)
        writer.append_data(image)