
import os
import numpy as np
import optimal_functions as of
import matplotlib.pyplot as plt
work_dir='/home/ariel2/Projects/optimal_mod_runs/'
os.chdir(work_dir)
from matplotlib import colors
import re
import pandas as pd

mod_id='sm_2'
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
mod_dir= os.path.join(work_dir,mod_id)
#%%
df_mod_runs,completed,failed=of.check_model_runs(work_dir,5)

#%% 
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
of.print_last_lines(os.path.join(mod_dir, f"results_sp10_cleaned.tec"))
#%%
ibound_arr_dir = os.path.join(mod_dir, 'ibound_arr.npy')
ibound_arr = np.load(ibound_arr_dir, allow_pickle = True)

#   get the nlay, nrow and ncol values
nlay, nrow, ncol = ibound_arr.shape[0], ibound_arr.shape[1], ibound_arr.shape[2]
sp=22 #Specifiy the stress period to be plotted

var_names=['X','Y','Z','HEAD','CONC','VX','VY','VZ']
results=os.path.join(mod_dir, f"results_sp{sp}_cleaned.tec")
df = pd.read_csv(results,skiprows=0,names=var_names)
print(df.head())



# Extract the X, Z, CONC, and HEAD columns
x = df['X'].values
z = df['Z'].values #convert to meters
conc = df['CONC'].values
head = df['HEAD'].values

# Determine the dimensions of the grid (assuming it's a regular grid)
nx = len(np.unique(x))
nz = len(np.unique(z))

# Reshape the CONC and HEAD arrays
conc_2d = conc.reshape((nz, nx))  # Reshape into a 2D array (Z x X)
head_2d = head.reshape((nz, nx))
conc_2d =np.where(conc_2d>36,np.nan,conc_2d)
head_2d =np.where(head_2d<-999,np.nan,head_2d)

# Create the plot
top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
base_elev=of.read_mod_file(mod_data,mod_id,'Toe of slope')
z_vals=np.linspace(top_elev,-base_elev,5)

plt.figure(figsize=(7,  2))
plt.imshow(conc_2d, extent=[x.min(),x.max(), -10*z.max(), 10*z.min()], 
           aspect='auto', vmin=0, vmax=36,cmap='jet')
plt.colorbar(label='CONC')
plt.xlabel('X')
plt.ylabel('Elevation (m)')
#plt.ylim(z.max(), z.min())
# Set yticks and yticklabels to match z_vals
plt.yticks(z_vals, labels=[f"{val:.1f}" for val in z_vals]) 
plt.title(f'Salinity Profile  - {mod_id} - sp {sp}')
#plt.savefig(f"{os.path.splitext(filename)[0]}_conc_profile.png")  # Save the plot
plt.show()

plt.figure(figsize=(7,  2))
plt.imshow(head_2d, extent=[x.min(), x.max(), -10*z.max(), 10*z.min()], 
           aspect='auto',cmap='viridis')
plt.colorbar(label='HEAD')
plt.xlabel('X')
plt.ylabel('Elevation (m)')
plt.yticks(z_vals, labels=[f"{val:.1f}" for val in z_vals])
plt.title(f'HEAD Profile - {mod_id} - sp {sp}')
#plt.ylim(z.max(), z.min())
#plt.savefig(f"{os.path.splitext(filename)[0]}_head_profile.png")  # Save the plot
plt.show()


# Add a colorbar to show porosity values
#cbar = fig.colorbar(im, ax=ax, shrink=0.6, label='Porosity')

# Show the plot
plt.show()