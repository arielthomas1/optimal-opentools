
import os
import numpy as np
import optimal_functions as of
import matplotlib.pyplot as plt
work_dir='/home/ariel2/Projects/optimal_mod_runs/'
os.chdir(work_dir)
from matplotlib import colors
import re

mod_id='sm_7'

mod_dir= os.path.join(work_dir,mod_id)
#%%
df_mod_runs,completed,failed=of.check_model_runs(work_dir,25)

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
    of.clean_tec_file(filename)
#%%


# Assuming your porosity array is named 'porosity' and has shape [294, 1, 1600]
# Load your porosity data here
# For example:
# porosity = np.load('path/to/your/porosity.npy')
np.random.seed(42) # for demonstration
porosity = np.random.rand(294, 1, 1600) * 0.4 + 0.1 # Example porosity values between 0.1 and 0.5

# Reshape the porosity array to 2D
porosity_2d = porosity.squeeze()

# Determine the extent of your model (assuming unit spacing)
extent = [0, porosity_2d.shape[1], porosity_2d.shape[0], 0]

# Create the plot
fig, ax = plt.subplots()

# Plot the porosity data using imshow
im = ax.imshow(porosity_2d, cmap='viridis', extent=extent, aspect='auto') # Using 'viridis' colormap

# Add labels and title
ax.set_xlabel('X-direction (Column)')
ax.set_ylabel('Z-direction (Row)')
ax.set_title('Porosity Distribution (X vs Z)')

# Add a colorbar to show porosity values
cbar = fig.colorbar(im, ax=ax, shrink=0.6, label='Porosity')

# Show the plot
plt.show()