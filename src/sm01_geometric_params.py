# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:07:35 2024

@author: Ariel

Preamble: 
 This script will be used to prepare the input parameter data space for building the surrogate models. 
 It will include statistical analysis and plotting of the distributions for each parameter input into the model
 Information on the type of distribution for each parameter such as skewness or kurtosis, which will influence the type of ML algorithms
 that will be suitable for subsequent data analysis.
 
 #List of parameters
 # sw (shelf width) - distance from coastline to shelf break
 # sbd (shelf break depth) - seafloor depth at shelf break
 # cust (coastal unconsolidated sediment thickness)
 # cst (coastal sediment thickness) - thickness of the sediment column at the coastline
 # tst (terrestrial sediment thickness) - thickness of sediment column at inland reference point
 # te (terrestrial elevation) - elevation of inland boundary of model
 # tos (toe of slope)

"""

#%% Importing Libraries and functions
import os
#work_dir=r"H:\My Drive\OPTIMAL\Project work\optimal"
work_dir='/home/ariel2/Projects/optimal'
os.chdir(work_dir)
import numpy as np
import seaborn as sns
import pandas as pd
import optimal_functions as of
import matplotlib.pyplot as plt
import math
import random
import ArchPy as ap
import geone
import geone.covModel as gcm
import sys
import pyvista as pv

from scipy.stats import norm
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator
# Set the seed
#np.random.seed(73)

#intializing parameter propery table
# List of parameters
parameters = ['sw', 'sbd','cust', 'cst', 'tst', 'te_10km', 'te_20km','te_30km','tos']
mean_vals = [110, 130, 300,150, 150, 700, 800, 900, 2500 ]
std_val = [30, 20, 100,10, 50, 50, 50, 50,  200 ]
skew = [3,0,0,0,0,0,0,0,0]
kurtosis = [8,3,3,3,3,3,3,3,3]
par_stats = pd.DataFrame({
    'parameter': parameters,
    'mean': mean_vals,
    'std_dev': std_val,
    'skew': skew,
    'kurt': kurtosis
})

#Windows setup
#input_data=r'H:\My Drive\OPTIMAL\Project work\optimal\surrogate_mode_tables' #input data
#output_data=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections" # main folder with all output data including figures
#mod_data=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections\surrogate_mod_summary" # text files containing all the model parameters

#Linux setup
input_data='/home/ariel2/Projects/optimal/surrogate_mode_tables'
output_data='/home/ariel2/Projects/optimal/surrogate_sections'
mod_data='/home/ariel2/Projects/optimal/surrogate_sections/surrogate_mod_summary'
fig_fol='/home/ariel2/Projects/optimal/surrogate_sections/figures'
#%% Coastal Sediment Thickness

# Importing coastal sediment thickness estimate dataset from csv table
df_cst_zam=pd.read_csv('{}/coastal_unconsol_thickness_zamrsky.csv'.format(input_data),delimiter=';')

# Replace commas with dots in all columns - because I am too lazy to update computer regional settings so that decimals are exported as points and not commas
df_cst_zam = df_cst_zam.map(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

# Convert columns to numeric where necessary
df_cst_zam = df_cst_zam.apply(pd.to_numeric)

#Quick sanity check 
df_cst_zam.describe()

#saving the actual variable of interest as a df for some pre-analysis

df_cst=df_cst_zam['ate_avg']
df_cst.describe()
print('Age: skewness is {} and Kurtosis is {}'.format(df_cst.skew(),df_cst.kurtosis()))
mean=df_cst.mean().round(1)
std=df_cst.std().round(1)

#Visulaizing the ditribution

fig2=plt.figure(figsize=(8, 4))


sns.kdeplot(df_cst, fill=True, label=f'- $\mu$: {mean} , $\sigma$: {std}')

# Add title and legend
plt.title('KDE of Coastal sediment thickness (Australia)')
plt.xlabel('Value (m)')
plt.ylabel('Density')
plt.legend()

# Show plot
plt.show()

#%% Coastal SEDIMENT THICKNESS
# Cleaning the dataset of nulll values and renaming
df_cst_zam.rename(columns={'GlobSedv2_Clip_ProjectRaster': 'GlobSed'}, inplace=True)
df_cst_cleaned=df_cst_zam[df_cst_zam['GlobSed'].notna()].copy()
#%% Finding the ratio of Unconsolidated sediments to sediment thickness
df_cst_cleaned['ate_to_sed']=-df_cst_cleaned['ate_avg']/df_cst_cleaned['GlobSed']
mean_val=df_cst_cleaned['ate_to_sed'].mean()

#%% extracting Zamrsky and GlobSed datasets to individual dfs
df_cst_zam=-df_cst_cleaned['ate_avg']
z_mean=df_cst_zam.mean().round(1)
z_std=df_cst_zam.std().round(1)

# writing stats to the master parameter dataframe
par_stats.loc[par_stats['parameter']=='cust','mean']=df_cst_zam.mean().round(1)
par_stats.loc[par_stats['parameter']=='cust','std_dev']=df_cst_zam.std().round(1)
par_stats.loc[par_stats['parameter']=='cust','skew']=df_cst_zam.skew().round(1)
par_stats.loc[par_stats['parameter']=='cust','kurt']=df_cst_zam.kurt().round(1)


df_cst_Glo=df_cst_cleaned['GlobSed']
g_mean=df_cst_Glo.mean().round(1)
g_std=df_cst_Glo.std().round(1)

# writing stats to the master parameter dataframe
par_stats.loc[par_stats['parameter']=='cst','mean']=df_cst_Glo.mean().round(1)
par_stats.loc[par_stats['parameter']=='cst','std_dev']=df_cst_Glo.std().round(1)
par_stats.loc[par_stats['parameter']=='cst','skew']=df_cst_Glo.skew().round(1)
par_stats.loc[par_stats['parameter']=='cst','kurt']=df_cst_Glo.kurt().round(1)
#Plot for comparison of distributions

fig3=plt.figure(figsize=(8, 4))

#Plotting Zam
sns.kdeplot(df_cst_zam, fill=True, label=f'ATE - $\mu$: {z_mean} , $\sigma$: {z_std}')
#Plotting Globsed
sns.kdeplot(df_cst_Glo, fill=True, label=f'GlobSed - $\mu$: {g_mean} , $\sigma$: {g_std}')


# Add title and legend
plt.title('KDE of Coastal sediment thickness (Australia)')
plt.xlabel('Value (m)')
plt.ylabel('Density')
plt.legend()

plt.xlim(-100, 3000)
# Show plot
plt.show()


#%% Terrestrial Elevation Data 
''' For elevation data, buffers were created at 10, 20 and 30 KM inland from the coast. Elevation
was then extracted from the DEM at every 5KM along the buffer lines to create the dataset and derive
statistical parameters'''

# Data from 10 KM Buffer
file_10km = os.path.join(input_data, "Australia_10KM_Buffer_Elevation.csv")
df_10km = pd.read_csv(file_10km, delimiter=';')
df_10km_clean = of.clean_null_values(df_10km, 'Australia_elevation')

# Data from 20 KM Buffer
file_20km = os.path.join(input_data, "Australia_20KM_Buffer_Elevation.csv")
df_20km = pd.read_csv(file_20km, delimiter=';')
df_20km_clean = of.clean_null_values(df_20km, 'Australia_elevation')

# Data from 30 KM Buffer
file_30km = os.path.join(input_data, "Australia_30KM_Buffer_Elevation.csv")
df_30km = pd.read_csv(file_30km, delimiter=';')
df_30km_clean = of.clean_null_values(df_30km, 'Australia_elevation')

# Compute statistics in one step
stats_10km = df_10km_clean['Australia_elevation'].agg(['mean', 'std', 'skew', 'kurt']).round(1)
stats_20km = df_20km_clean['Australia_elevation'].agg(['mean', 'std', 'skew', 'kurt']).round(1)
stats_30km = df_30km_clean['Australia_elevation'].agg(['mean', 'std', 'skew', 'kurt']).round(1)

# Update par_stats in a single operation
par_stats.loc[par_stats['parameter'] == 'te_10km', ['mean', 'std_dev', 'skew', 'kurt']] = stats_10km.values
par_stats.loc[par_stats['parameter'] == 'te_20km', ['mean', 'std_dev', 'skew', 'kurt']] = stats_20km.values
par_stats.loc[par_stats['parameter'] == 'te_30km', ['mean', 'std_dev', 'skew', 'kurt']] = stats_30km.values

# writing stats to the master parameter dataframe
# par_stats.loc[par_stats['parameter']=='te_10km','mean']=df_10km_clean['Australia_elevation'].mean().round(1)
# par_stats.loc[par_stats['parameter']=='te_10km','std_dev']=df_10km_clean['Australia_elevation'].std().round(1)
# par_stats.loc[par_stats['parameter']=='te_10km','skew']=df_10km_clean['Australia_elevation'].skew().round(1)
# par_stats.loc[par_stats['parameter']=='te_10km','kurt']=df_10km_clean['Australia_elevation'].kurt().round(1)
par_stats

#%% Visualizing the distribution of surrogate model input parameters

# Create a figure and axes
fig1=plt.figure(figsize=(10, 6))

# Initialize the x-axis range to accommodate all distributions
min_value = float('inf')
max_value = float('-inf')
#df_par_stats = par_stats.loc[par_stats['parameter'] != 'tos']
df_par_stats = par_stats.loc[(par_stats['parameter'] != 'tos') & (par_stats['parameter'] != 'cst')]

# Loop through each row of the DataFrame to calculate distributions
# for index, row in df_par_stats.iterrows():
#     parameter = row['parameter']
#     mean = row['mean']
#     std_dev = row['std_dev']
    
#     # Generate data for the distribution based on mean and std_dev extracted from global datasets
#     distribution_data = np.random.normal(loc=mean, scale=std_dev, size=10000)
    
#     # Plot the KDE on the same axes, including the mean and std. dev. of each parameter in the legend
#     sns.kdeplot(distribution_data, fill=True, label=f'{parameter} - $\mu$: {mean} , $\sigma$: {std_dev}')
    
#     # Adjust the range for the x-axis to fit all distributions
#     min_value = min(min_value, mean - 4 * std_dev)
#     max_value = max(max_value, mean + 4 * std_dev)

# # Set the x-axis limits based on the largest distribution
# plt.xlim(min_value, max_value)

# # Add title and legend
# plt.title('KDE of Surrogate model input parameters')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
from scipy.stats import gamma
for index, row in df_par_stats.iterrows():
    parameter = row['parameter']
    mean = row['mean']
    std_dev = row['std_dev']
    skewness = row['skew']

    # Approximate Gamma parameters (this is a simplification, more robust methods exist)
    if skewness > 0:  # Gamma distribution is often used for positive skew
        alpha = 4 / skewness**2  # shape parameter
        beta = mean / alpha       # scale parameter
        if alpha > 0 and beta > 0:
            distribution_data = gamma.rvs(a=alpha, scale=beta, size=20000)
            sns.kdeplot(distribution_data, fill=True, label=f'{parameter} - $\mu$:{mean:.2f}, $\sigma$:{std_dev:.2f}, Sk:{skewness:.2f}')

            min_value = min(min_value, np.min(distribution_data))
            max_value = 600#max(max_value, np.max(distribution_data))
        else:
            print(f"Warning: Invalid Gamma parameters for {parameter}")
    else:
        # Fallback to clipping normal distribution if skewness is not positive
        distribution_data = np.random.normal(loc=mean, scale=std_dev, size=10000)
        distribution_data[distribution_data < 0] = 0
        sns.kdeplot(distribution_data, fill=True, label=f'{parameter} - $\mu$:{mean:.2f}, $\sigma$:{std_dev:.2f}, Sk:{skewness:.2f}')
        min_value = min(min_value, np.min(distribution_data))
        max_value = 600#max(max_value, np.max(distribution_data))

plt.xlim(min_value, max_value)
plt.title('KDE of Surrogate model input parameters (with Skewness)',fontsize=18)
plt.xlabel('Value (m)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.legend(fontsize=12)
plt.show()

# Show plot
plt.show()
fig1.savefig(f'{fig_fol}/KDE_input_params_corrected.png', dpi=450, bbox_inches='tight')

#%% Plotting distributions

mean=[df_10km_clean['Australia_elevation'].mean().round(1),df_20km_clean['Australia_elevation'].mean().round(1),df_30km_clean['Australia_elevation'].mean()]
std=[df_10km_clean['Australia_elevation'].std(),df_20km_clean['Australia_elevation'].std(),df_30km_clean['Australia_elevation'].std()]


#Plot for comparison of distributions

fig3=plt.figure(figsize=(6, 3))

#Plotting 10 KM
sns.kdeplot(df_10km_clean['Australia_elevation'], fill=True, label=f'10 KM - $\mu$: {mean[0]:.1f} , $\sigma$: {std[0]:.1f}')
#Plotting 20 KM
sns.kdeplot(df_20km_clean['Australia_elevation'], fill=True, label=f'20 KM- $\mu$: {mean[1]:.1f} , $\sigma$: {std[1]:.1f}')
#Plotting 30 KM
sns.kdeplot(df_30km_clean['Australia_elevation'], fill=True, label=f'30 KM- $\mu$: {mean[2]:.1f} , $\sigma$: {std[2]:.1f}')

# Add title and legend
plt.title('KDE of Inland Elevation (Australia)')
plt.xlabel('Value (m)')
plt.ylabel('Density')
plt.legend()


#%% Extracting Model parameters 
''' MOdel top and base will be determined by extracting a random value for each parameter
extracted from the distributions visualized in the previous step'''

#TODO Create function to make base of surrogate model using random values of inputs
#TODO Create function to make top of surrogate model using random values from input parameter space 
#TODO Develop a methodology to QC the shape of the model create and cast out unrealistic creations

# Set the total number of surrogate models to generate
num_models = 10

# Dictionary to store seeds for reproducibility
seeds_dict = {}

# Loop to create surrogate models
for i in range(1, num_models + 1):
    SEED = np.random.randint(2569)
    rng = np.random.default_rng(SEED)
    mod_id = f'sm_{i}'
    
    # Store the seed in the dictionary
    seeds_dict[mod_id] = SEED

    # # Save seeds to a file for reproducibility
    # with open("r{}\seeds.txt".format(output_data), "w") as file:
    #     for mod_id, seed in seeds_dict.items():
    #         file.write(f"{mod_id}: {seed}\n")
    
    #Model dimensions
    
    # INLAND EXTENT
    inland_sect=10000
    #--------------
    # TOP ELEVATION
    #retrieving random val based on the mean and std dev from the parameters table
    top_elev = of.get_random_top_elev(of.get_mean, of.get_sdev, 'te_10km', par_stats, rng)
    #TODO - check and add skewness from real data
    #--------------
    # COAST SEDIMENT THICKNESS
    #retrieving random val based on the mean and std dev from the parameters table
    cst = int(0.7*of.get_random_cst(of.get_mean, of.get_sdev, 'cst', par_stats, rng))
    #--------------
    # COASTAL UNCONSOLIDATED SEDIMENT THICKNESS - #ATE - Zamrsky
    cust = of.get_random_cust(of.get_mean, of.get_sdev, 'cust', par_stats, rng)
    #--------------
    # TERRESTRIAL SEDIMENT THICKNESS
    #retrieving random val based on the mean and std dev from the parameters table
    # tst=np.round(norm.rvs(
    #                 loc=of.get_mean('tst',par_stats), 
    #                 scale=of.get_sdev('tst',par_stats),
    #                 random_state=rng.integers(10000)),-1)
    
    #--------------
    #SHELF WIDTH
    #retrieving random val based on the mean and std dev from the parameters table
    sw = of.get_shelf_width(of.get_mean, of.get_sdev, 'sw', par_stats, rng)
    #--------------
    #SHELF EDGE SEDIMENT THICKNESS
    #retrieving random val based on the mean and std dev from the parameters table
    sbd = of.get_shelf_edge_thickness(of.get_mean, of.get_sdev, 'sbd', par_stats, rng)
    #--------------
    #TOE OF SLOPE
    #retrieving random val based on the mean and std dev from the parameters table
    tos = of.get_tos_value(of.get_mean, of.get_sdev, 'tos', par_stats, rng)
    #--------------
    # OTHER SUPPORTING PARAMETERS
    tst=int(0.66*cst)
    slope_width=random.randint(15000, 20000) # hor. distance between shelf break and toe of slope in m
    mod_len=inland_sect+int(sw)+slope_width # Defining the simulation grid size. 
    unconsol_ratio=cust/cst
    #--------------
    # Calculating Average slope angle in degrees
    slope_angle = of.calculate_slope_angle(sbd, sw)
    #--------------
    # Finding the anchor point at the shelf break to form a straight base line
    base_gradient=(-tos--cst)/(sw+slope_width)
    intercept=-cst-base_gradient*0
    base_angle = np.round(math.degrees(math.atan(base_gradient)))
    z_sb_anchor=base_gradient*(sw)+intercept
    
    shelf_anchor_base=np.minimum(tos,(2*cst))
    #--------------
    #Defining anchor points
    x_mod = np.array([0, int(inland_sect), int(inland_sect + sw), int(mod_len)])
    z_top = np.array([top_elev, 0, -sbd, -tos])
    z_base = np.array([top_elev - tst, -cst, z_sb_anchor, -tos])
    #z_base_2=np.array([z_a_base,z_b_base,z_sb_anchor,z_d_base])
    #--------------
    #Defining well total depths
    td_a=top_elev+cst
    td_b=cst +500  #arbitrary buffer
    td_c=td_d=(tos-sbd)-5 #arbitrary buffer
    #--------------
    #creating control wells at 10 km intervals
    x_bhs=np.arange(0,sw,10000)
    z_bhs=np.ones_like(x_bhs)
    plt.plot(x_mod/1000, z_top,'-o', label='Top Model')      # Interpolated spline
    plt.plot(x_mod/1000, z_base,'-o', label='Base Model')
    plt.scatter(x_bhs/1000,z_bhs,marker='x', label='Well locations')
    #plt.plot(x_mod/1000, z_base_2,'-x', label='Base Model')
    
    plt.legend()
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (m)")
    plt.title(f"Control Points QC Sanity Check - {mod_id}")
    plt.grid()
    plt.show()
    # Creating synthentic Boreholes to Anchor surrogate model surfaces.
    '''Four boreholes will be created for each SM realization. 
    The boreholes a, b, c and d will represent the control points at the
     inland mark, coast, shelf break and toe of slope, respectively.'''
    bh_folder="/home/ariel2/Projects/optimal/surrogate_sections/surrogate_boreholes"

    with open(r"{}/bh_{}.lbh".format(bh_folder,mod_id),"w") as file:
        file.write('bh_ID,bh_x,bh_y,bh_z,bh_depth\n')
        file.write('a,1,50,{},{}\n'.format(top_elev/10,td_a/10)) # inland anchor point
        file.write('b,{},50,5,{}\n'.format(inland_sect/100,td_b/10)) # coastal anchor point
        file.write('c,{},50,{},{}\n'.format((inland_sect+sw)/100,sbd/10,td_c/10)) # shelf anchor point
        #file.write('d,{},50,{},{}\n'.format((inland_sect+sw+slope_width-50)/100,sbd/10,td_d/10)) # slope anchor point
    #Sanity checks
    #TODO Add sanity checks for values to ensure they honour realistic geometry
    
    #Generating list of unit data in boreholes
    with open(r"{}/bh_{}.ud".format(bh_folder,mod_id),"w") as file:
        file.write('bh_ID,Strat,top,bot\n')
        file.write('a,U,{},{} \n'.format(top_elev/10,(top_elev-tst)/10)) # inland anchor point
        file.write('a,L,{},{} \n'.format((top_elev-tst)/10,-tos/10))
        file.write('b,U,-5,{} \n'.format(-cst/10)) # coastal anchor point
        file.write('b,L,{},{} \n'.format(-cst/10,-tos/10))
        file.write('c,U,{},{} \n'.format(-sbd/10,-shelf_anchor_base/10)) # shelf anchor point
        file.write('c,L,{},{} \n'.format(-shelf_anchor_base/10,-tos/10))
        #file.write('d,U,{},{} \n'.format((-tos/10)-10,-tos/10)) # slope anchor point
        #file.write('d,L,{},{} \n'.format(-tos/10,-tos/10))
        
    #Generating list of facies data in boreholes
    with open(r"{}/bh_{}.fd".format(bh_folder,mod_id),"w") as file:
        file.write('bh_ID,facies_ID,top,bot\n')
        file.write('a,silt,{},{} \n'.format(top_elev/10,(top_elev-tst)/10)) # inland anchor point
        file.write('a,rock,{},{} \n'.format((top_elev-tst)/10,-tos/10))
        file.write('b,silt,-5,{} \n'.format(-cst/10)) # coastal anchor point
        file.write('b,rock,{},{} \n'.format(-cst/10,-tos/10))
        file.write('c,silt,{},{} \n'.format(-sbd/10,-shelf_anchor_base/10)) # shelf anchor point
        file.write('c,rock,{},{} \n'.format(-shelf_anchor_base/10,-tos/10))
        #file.write('d,silt,{},{} \n'.format(-tos/10,-tos/10)) # slope anchor point
        #file.write('d,rock,{},{} \n'.format(-tos/10,-tos/10))
        
    # Defining path to bh files and data
    bh_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.lbh'.format(mod_id)))
    units_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.ud'.format(mod_id)))
    facies_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.fd'.format(mod_id)))
    
    with open(r"{}/surrogate_mod_summary/{}.txt".format(output_data,mod_id), "w") as file:
        file.write('Model parameter summary (m): \n Inland distance = {} \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Toe of slope = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total model length = {} \n Top elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}\n Unconsol. ratio = {} \n\nAquifer dimsensions (m): \n'.format(inland_sect,sw,slope_width,sbd,tos,slope_angle,inland_sect,mod_len,top_elev,tst,cst,cust/cst)) 
    file.close()

#print('Model parameter summary (m): \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total Model length = {} \n Top Elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}\n Unconsol. ratio = {}'.format(sw,slope_width,sbd,slope_angle,inland_sect,mod_len,top_elev,tst,cst,cust/cst))

# Save seeds to a file for reproducibility
with open(r"{}/seeds.txt".format(output_data), "w") as file:
    for mod_id, seed in seeds_dict.items():
        file.write(f"{mod_id}: {seed}\n")
        


#%%
num_models=10
lbh_header = ['bh_ID', 'bh_x', 'bh_y', 'bh_z', 'bh_depth']
fd_header=['bh_ID','facies_ID','top','bot']
ud_header=['bh_ID','Strat','top','bot']

df_lbh=pd.DataFrame(columns=lbh_header)
df_fd=pd.DataFrame(columns=fd_header)
df_ud=pd.DataFrame(columns=ud_header)

# Initialize dictionary to store ArchPy model objects
mod_dict = {}

# Loop to generate multiple ArchPy objects
for i in range(1, num_models+1):
    SEED = np.random.randint(239)
    rng = np.random.default_rng(SEED)
    mod_id = f'sm_{i}'  # Generate model ID
    print(mod_id)
    # Create ArchPy object and store it in dictionary
    mod_dict[mod_id] = ap.base.Arch_table(mod_id, "{}".format(output_data), seed=777, verbose=1)

    '''The simulation grid will be designed such that all the modesl have the same cell size
    despite having different dimensions. The number of cells (nx) will be a function 
    of the model length (mod_len) i.e., nx=mod_len/100'''
    #Reading the model parameters from the model summary files
    mod_len=of.read_mod_file(mod_data, mod_id, 'Total model length')
    
    top_elev=of.read_mod_file(mod_data,mod_id, 'Top elevation')
    tos=of.read_mod_file(mod_data,mod_id, 'Toe of slope')
    # Defining the simulation grid size and adding grid. 
    sx=sy=100 #cell size in x,y
    sz=10  #cell size in z
    nx=int(mod_len/sx) #no. of cells in x
    ny=1 
    nz=int((top_elev+tos)/sz)
    ox=oy=0
    oz=-tos 
    #oz=top_elev #
    dimensions= (nx,ny,nz)
    spacing = (sx, sy, sz)
    origin = (ox, oy, oz)

#------------------------------------------------------------------
# MODEL GEOMETRY 
#------------------------------------------------------------------
   #%Creating interpolated Top and Base Surfaces 

    #Defining the aquifer geometries using a fixed value relative to model bounds
    inland_sect=of.read_mod_file(mod_data,mod_id, 'Inland length') #Extracting model inland length from summary file
    sbd=of.read_mod_file(mod_data,mod_id, 'Shelf break depth') #Extracting model shelf break depth from summary file
    sw=of.read_mod_file(mod_data,mod_id, 'Shelf width') #Extracting model shelf width from summary file
    #TODO Add cross-correlation constraints on aquifer dimensions
    min_aq=int(inland_sect+2000) # defining a minium offshore extent for aquifer in surrogate models
    aq_len=random.randint(min_aq, int(sw)) # length of the aquifer
    ob_thickness=random.randint(0,250) #thickness of the overburden layer or depth to top of aq.
    aq1_thickness=random.randint(200,600) #thickness of aquifer itself
    
    # Write the aquifer properties to the model summary file
    with open(r"{}/surrogate_mod_summary/{}.txt".format(output_data,mod_id), "a") as file:
        file.write(' Aquifer length = {} \n Overburden thickness = {} \n Aquifer1 thickness = {} \n'.format(aq_len,ob_thickness,aq1_thickness))
    file.close()


    #re-creating X and Z arrays based on model realization
    x_mod = np.array([0, int(inland_sect), int(inland_sect + sw), int(mod_len)])
    z_top = np.array([top_elev, 0, -sbd, -tos])
    
    x_new = np.linspace(x_mod.min(), x_mod.max(), nx)
    #TODO Modify the number of boreholes to correspond with the aquifer offshore extent
    x_bhs = np.linspace(sx,sw,int(nx/100))
    #modify the well locations according to the randomly allocated aquifer length
    x_bhs_aquifer = x_bhs#[x_bhs<=aq_len]

    # Create a cubic B-spline 
    spline_top = PchipInterpolator(x_mod, z_top)
    z_top_new = spline_top(x_new)

    spline_base = PchipInterpolator(x_mod, z_base)  
    z_base_new = spline_base(x_new)

    #Creating boreholes
    z_bhs=spline_top(x_bhs_aquifer)
    z_unit_base=spline_base(x_bhs_aquifer)
    #setting the the total depth of the boreholes equal to the model domain
    td_bhs=abs(spline_base(x_bhs_aquifer)-z_bhs)
    
#---------------------------------------------------------
# AQUIFER SHAPE AND THICKNESS
#---------------------------------------------------------
    #Defining top of aquifer based on thickness of overburden confining layer
    
    # Step 1: Define borehole locations
    x_bhs = np.linspace(sx, sw+inland_sect, int(nx/100))
    x_bhs_aquifer = x_bhs#[aq_mask]
    # Step 2: Compute surfaces at borehole locations
    z_bhs = spline_top(x_bhs)
    z_unit_base = spline_base(x_bhs)
    
    
    # Step 3: Apply pinchout taper function
    x_pinchoff = aq_len
    steepness = 12
    midpoint = x_pinchoff * 0.95
    
    taper_factor = 1 / (1 + np.exp(steepness * (x_bhs - midpoint) / x_pinchoff))
    top_shift = aq1_thickness * (1 - taper_factor)
    # Define a custom dip for geological realism
    dip_rate = 0.0015  # meters dip per horizontal meter (e.g. 2%)
    
    bot_aq1 = spline_top(x_bhs) - (ob_thickness + aq1_thickness)
    # Apply dip increasing with distance
    bot_aq1 = bot_aq1 - dip_rate * (x_bhs - x_bhs[0])
    
    top_aq1 = bot_aq1 + (aq1_thickness * taper_factor)
    

    top_ob1=spline_top(x_bhs_aquifer)
    bot_ob1=top_aq1
    bot_sed_column=spline_base(x_bhs_aquifer)
  
    #------------------------------------------------------------------
    # BOREHOLES, UNITS & FACIES
    #------------------------------------------------------------------
    
    # Create the list of borehole names
    borehole_names = [f"bh_{i}" for i in range(1, len(x_bhs_aquifer) + 1)]
    # Ensure df_lbh is always initialized with the correct number of rows
    df_lbh = pd.DataFrame(index=range(len(borehole_names)))  
    #Fill boreholes
    df_lbh = df_lbh.assign(
        bh_ID=borehole_names,
        bh_x=np.round(x_bhs_aquifer),
        bh_y=(sy / 2) * np.ones_like(x_bhs_aquifer),
        bh_z=np.floor(z_bhs),
        bh_depth=np.floor(td_bhs)
    )

    #print(df_lbh) #for QC
    ### Adding stratigraphic units for ArchPy
    # # Step 1: Create dummy boreholes past the pinchout
    n_dummy = 3
    x_dummy = np.linspace(x_bhs_aquifer[-1] + 50, x_bhs_aquifer[-1] + 2000, n_dummy)
    z_dummy = spline_top(x_dummy)
    base_dummy = spline_base(x_dummy)
    top_ob_dummy = spline_top(x_dummy)
    bot_ob_dummy = top_ob_dummy-(aq1_thickness+ob_thickness) - dip_rate * (x_dummy - x_bhs[0]) #calculating the extrapolated base of the overburden
    bot_sed_dummy = base_dummy
    
    # Step 2: Add to borehole list
    dummy_names = [f"dbh_{i}" for i in range(1, n_dummy + 1)]
        
        # Create a new DataFrame for dummy boreholes
    df_dummy = pd.DataFrame({
        "bh_ID": dummy_names,
        "bh_x": np.round(x_dummy),
        "bh_y": (sy / 2) * np.ones_like(x_dummy),
        "bh_z": np.floor(z_dummy),
        "bh_depth": np.floor(abs(base_dummy - z_dummy))
    })
    
    # Concatenate with existing boreholes
    df_lbh = pd.concat([df_lbh, df_dummy], ignore_index=True)
    
    df_lbh.to_csv(r"{}/surrogate_boreholes/bh_{}.lbh".format(output_data,mod_id),index=False)
    # Start index
    row_idx = 0
    
    # Loop to populate the DataFrame
    for bh_id, sand_top, sand_bot in zip(borehole_names, top_aq1, bot_aq1):
        df_ud.loc[row_idx] = [bh_id, 'aq1', np.floor(sand_top),np.floor(sand_bot)]  # Add values row by row
        row_idx += 1
        
    for bh_id, ob_top, ob_bot in zip(borehole_names, top_ob1, bot_ob1):
        df_ud.loc[row_idx] = [bh_id, 'ob1', np.floor(ob_top),np.floor(ob_bot)]  # Add values row by row
        row_idx += 1
    # This loop populates the df with the remaining sedimentary column interval from the base of the aquifer layer 
    # to the basal surface of the model domain.     
    for bh_id, sed_column_top, sed_column_bot in zip(borehole_names, bot_aq1, bot_sed_column):
        df_ud.loc[row_idx] = [bh_id, 'sed_col', np.floor(sed_column_top),np.floor(sed_column_bot)]  # Add values row by row
        row_idx += 1
            
    #     # Step 3: Append to df_ud (continue row_idx from before)
    for bh_id, ob_top, ob_bot in zip(dummy_names, z_dummy, bot_ob_dummy):
        df_ud.loc[row_idx] = [bh_id, 'ob1', np.floor(ob_top), np.floor(ob_bot)]
        row_idx += 1
    
    for bh_id, sed_top, sed_bot in zip(dummy_names, bot_ob_dummy, base_dummy):
        df_ud.loc[row_idx] = [bh_id, 'sed_col', np.floor(sed_top), np.floor(sed_bot)]
        row_idx += 1
       
    #writing to df
    df_ud.to_csv(r"{}/surrogate_boreholes/bh_{}.ud".format(output_data,mod_id),index=False)
    #print(df_ud) #for QC

    #Adding Facies to units
    
    row_idx = 0
    # Loop to populate the DataFrame
    for bh_id, sand_top, sand_bot in zip(borehole_names, top_aq1, bot_aq1):
        df_fd.loc[row_idx] = [bh_id, 'sand', np.floor(sand_top),np.floor(sand_bot)]  # Add values row by row
        row_idx += 1
        
    for bh_id, ob_top, ob_bot in zip(borehole_names, top_ob1, bot_ob1):
        df_fd.loc[row_idx] = [bh_id, 'clay', np.floor(ob_top),np.floor(ob_bot)]  # Add values row by row
        row_idx += 1
    # This loop populates the df with the remaining sedimentary column interval from the base of the aquifer layer 
    # to the basal surface of the model domain.     
    for bh_id, sed_column_top, sed_column_bot in zip(borehole_names, bot_aq1, bot_sed_column):
        df_fd.loc[row_idx] = [bh_id, 'silt', np.floor(sed_column_top),np.floor(sed_column_bot)]  # Add values row by row
        row_idx += 1
        # Add facies (NO aquifer/sand)
    for bh_id, ob_top, ob_bot in zip(dummy_names, z_dummy, bot_ob_dummy):
        df_fd.loc[row_idx] = [bh_id, 'clay', np.floor(ob_top), np.floor(ob_bot)]
        row_idx += 1
    
    for bh_id, sed_top, sed_bot in zip(dummy_names,bot_ob_dummy, base_dummy):
        df_fd.loc[row_idx] = [bh_id, 'silt', np.floor(sed_top), np.floor(sed_bot)]
        row_idx += 1
        
    df_fd.to_csv(r"{}/surrogate_boreholes/bh_{}.fd".format(output_data, mod_id), index=False)  
    
    # last_bh=f'bh_{len(x_bhs_aquifer)}'
    # # Update Units dataframe
    # #Select row where bh_ID == 'bh_7' and Strat == 'aq1'
    # mask = (df_ud['bh_ID'] == last_bh) & (df_ud['Strat'] == 'aq1') # Find the aquifer section of the last well 
    # df_ud.loc[mask, 'Strat'] = 'ob1' #Replace it with the overburden
    # #df_ud.loc[mask, 'bot']+=-50 # trying to force the erosion to truncate the layer
    
    # #Update Facies dataframe
    # mask = (df_fd['bh_ID'] == last_bh) & (df_fd['facies_ID'] == 'sand') # Find the aquifer section of the last well 
    # df_fd.loc[mask, 'facies_ID'] = 'clay' #Replace it with the overburden
    # #df_fd.loc[mask, 'bot']+=-50
    
    df_ud.to_csv(r"{}/surrogate_boreholes/bh_{}.ud".format(output_data,mod_id),index=False)
    df_fd.to_csv(r"{}/surrogate_boreholes/bh_{}.fd".format(output_data,mod_id),index=False)
    
    
    # Plotting for QC
    fig, ax = plt.subplots()
    
    # Fill the area between the two lines
    ax.fill_between(x_new/1000, z_top_new, z_base_new, color='orange', alpha=0.7)
    
    plt.scatter(x_bhs_aquifer/1000,z_bhs,marker='s',s=10,color='black',label='Boreholes')
    plt.scatter(x_bhs_aquifer/1000,-td_bhs,marker='s',s=10,color='black',label='Boreholes TD')
    #Markers
    plt.scatter(x_mod/1000, z_top, marker='x', color='gray',label='Control points')  # Original control points
    plt.scatter(x_mod/1000, z_base, marker='x', color='gray')  
    #Surfaces
    plt.plot(x_new/1000, z_top_new, color='blue',label='Top model int')      # Interpolated spline
    plt.plot(x_new/1000, z_base_new, color='black',label='Base model int')      # Interpolated spline
    plt.plot(x_bhs_aquifer/1000,top_aq1,color='yellow',label= 'Aquifer 1')
    plt.plot(x_bhs_aquifer/1000,bot_aq1,color='yellow')
    ax.fill_between(x_bhs_aquifer/1000, top_aq1, bot_aq1, color='yellow', alpha=0.7)
    
    plt.legend()
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (m)")
    plt.title("{} - Interpolation - QC ".format(mod_id))
    plt.show()
    
    fig.savefig('{}/figures/{}_geometry.png'.format(output_data,mod_id), dpi=450, bbox_inches='tight')


    # Creating top and base model arrays for input into model
    top_surf=np.array([z_top_new])
    bot_surf=np.array([z_base_new])
#adding grid
    mod_dict[mod_id].add_grid(dimensions, spacing, origin, top=top_surf, bot=bot_surf) 

    ### COVARIANCE MODELS ####
    #Covariance model for top surface
    covmodel_er = gcm.CovModel2D(elem=[('spherical', {'w':9.9, 'r':[30000,100]}),
                                       ('nugget',{'w':0.1})],
                                        alpha=-slope_angle)
    #COvariance model for sedimentary unit
    covmod_U=gcm.CovModel3D(elem=[('spherical', {'w':9.9, 'r':[30000,100,100]}),
                                  ('nugget',{'w':0.1})],
                                    gamma=-slope_angle)

### SURFACES ####

    #defining top model surface objects
    top=ap.base.Surface(name='top_mod',
                         dic_surf={'N_transfo': False, 'covmodel': covmodel_er, 'int_method': 'grf_ineq'},
                         contact='erode')


    #defining top aquifer 1 base surface object
    top_aq1=ap.base.Surface(name='top_aq1',
                         dic_surf={'N_transfo': False, 'covmodel': covmodel_er, 'int_method': 'grf_ineq'},
                         contact='erode')

    bot_aq1=ap.base.Surface(name='bot_aq1',
                         dic_surf={'N_transfo': False, 'covmodel': covmodel_er, 'int_method': 'grf_ineq'},
                         contact='erode')
#### UNITS AND FACIES #####
    cov_mod_sand=gcm.CovModel3D(elem=[('spherical',{'w':slope_angle,'r':[30000,100,100]})],
                                 alpha=0,beta=-slope_angle,gamma=0)
    cov_mod_shale=gcm.CovModel3D(elem=[('spherical',{'w':slope_angle,'r':[30000,100,100]})],
                                 alpha=0,beta=-slope_angle,gamma=0)
    #defining overburden
    ob_facies={'f_method':'homogenous','f_covmodel':None}
    ob1=ap.base.Unit(name='ob1',
                     order=1,
                     color='brown',
                     surface=top,
                     dic_facies=ob_facies)

    # Stochastic version
    # defining aquifer unit
    # aq_facies={'f_method':'SIS','f_covmodel':cov_mod_sand,'probability':[0.8,0.2]}
    
    # Homogenous version # Testing this option to improve numerical model stability  
    # defining aquifer unit
    aq_facies={'f_method':'homogenous','f_covmodel':None}
    aq1=ap.base.Unit(name='aq1',
                     order=2,
                     color='yellow',
                     surface=top_aq1,
                     dic_facies=aq_facies)
    



    # # Sochastic version - defining underlying unit
    # sed_col_facies={'f_method':'SIS',
    #                 'f_covmodel':[cov_mod_sand,cov_mod_sand,cov_mod_shale],
    #                  'probability':[0.2,0.4,0.4]}
    # # Homogenous version - defining underlying unit
    sed_col_facies={'f_method':'homogenous','f_covmodel':None}
    
    sed_col=ap.base.Unit(name='sed_col',
                     order=3,
                     color='darkorange',
                     surface=bot_aq1,
                     dic_facies=sed_col_facies)
    #defining subpile
    seq_strat=ap.base.Pile(name='sequence_strat')
    #Adding the aquifer unit to the subpile
    seq_strat.add_unit([sed_col,aq1,ob1])
    #Unit 1 -  unit representing the total sedimentary column of the shelf
    #NO covmodel defined at this stage because the model domain is being expicitly defined as homogeneous.
    
    # U_facies={'f_method':'SIS','f_covmodel':covmod_U}
    # U=ap.base.Unit(name= 'sed_lyr',
    #                 order=1,
    #                 color='brown',
    #                 surface=top,
    #                 dic_facies=U_facies)   
    

    U_facies={'f_method':'SubPile', 'SubPile':seq_strat}
    U=ap.base.Unit(name= 'shelf_lyr',
                    order=1,
                    color='brown',
                    surface=top,
                    ID=4,
                    dic_facies=U_facies)
    #TODO Add subpiles to include the aquifer interval as an explicit unit in the model
    # Sub unit 1- Aquifer unit embedded in shelf


    #Stratigraphic Pile
    strat_pile=ap.base.Pile(name='Strat')
    strat_pile.add_unit([U])
    mod_dict[mod_id].set_Pile_master(strat_pile)


#ADDING SURFACES TO THE STRAT PILE


    #Adding Facies
    sand=ap.base.Facies(ID=1, name='sand', color='yellow')
    silt= ap.base.Facies(ID=2, name='silt',color='darkorange')
    clay=ap.base.Facies(ID=3, name='clay',color='brown')
    gravel=ap.base.Facies(ID=4, name='gravel',color='palegreen')
    rock=ap.base.Facies(ID=5, name='rock', color='darkgray' )

    # # Geostatistical version 
    # ''' i.e. Each unit consists of multiple facies determined by stochastic model'''
    # aq1.add_facies([sand,silt])
    # ob1.add_facies([clay])
    # sed_col.add_facies([sand,silt,clay])
    # U.add_facies([sand,silt,clay])
    #U.add_facies([sand])
    # Homoegenous facies version
    ''' i.e. each unit only consits of one facies type'''
    aq1.add_facies([sand])
    ob1.add_facies([clay])
    sed_col.add_facies([silt])
    U.add_facies([sand,silt,clay])
    #Adding properties
    '''Porosity typically exhibits a gaussian normal distribution so this type
    of model will be assigned.'''
    
    cov_mod_por=gcm.CovModel3D(elem=[('gaussian',{'w':9.7,'r':[nx*sx/3,100,aq1_thickness/2]}),
                                     ('nugget', {'w':0.3}) ],
                                 alpha=0,beta=0,gamma=-slope_angle)
    
    cov_mod_k=gcm.CovModel3D(elem=[('gaussian',{'w':9.7,'r':[nx*sx/3,100,aq1_thickness/2]}),
                                   ('nugget', {'w':0.3}) ],
                                 alpha=0,beta=0,gamma=-slope_angle)
    
    cov_mod_clay=gcm.CovModel3D(elem=[('gaussian',{'w':9.7,'r':[nx*sx/2,100,aq1_thickness/2]}),
                                      ('nugget', {'w':0.3}) ],
                                 alpha=0,beta=0,gamma=-slope_angle)
    mean_vals_por=[0.3,0.45,0.6]
    mean_vals_k=[5,1,0.1]
    list_facies=[sand,silt,clay]

     # STOCHASTIC
    # por=ap.base.Prop(name="Por",facies=list_facies,
    #                  covmodels=[cov_mod_por,cov_mod_por,cov_mod_clay],
    #                  means=mean_vals_por,
    #                  int_method="sgs",
    #                  vmin=0.29,
    #                  vmax=0.61)
    
    # hyd_con=ap.base.Prop(name='K',facies=list_facies,
    #                      covmodels=[cov_mod_k,cov_mod_k,cov_mod_clay],
    #                      means=mean_vals_k,
    #                      int_method='sgs',
    #                      vmin=0.1,
    #                      vmax=5.5)
    # # HOMOEGENOUS
    por=ap.base.Prop(name="Por",facies=list_facies,
                     covmodels=[cov_mod_por,None,None],
                     means=mean_vals_por,
                     int_method=["sgs","homogenous","homogenous"],
                     vmin=0.20,
                     vmax=0.65)
    
    hyd_con=ap.base.Prop(name='K',facies=list_facies,
                         covmodels=[cov_mod_k,None,None],
                         means=mean_vals_k,
                         int_method=["sgs","homogenous","homogenous"],
                         vmin=0.1,
                         vmax=6)

    #Adding porosity to model object'
    mod_dict[mod_id].add_prop(por)
    mod_dict[mod_id].add_prop(hyd_con)

    #%
    #Reading  borehole files
    bh_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.lbh'.format(mod_id)))
    units_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.ud'.format(mod_id)))
    facies_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.fd'.format(mod_id)))
    
    # Importing Borehole data
    db,l_bhs=ap.inputs.load_bh_files(list_bhs=bh_path,
                                   units_data=units_path,
                                   facies_data=facies_path,
                                   altitude=True)

    #db

#Extracting boreholes
    boreholes=ap.inputs.extract_bhs(df=db, list_bhs=l_bhs,ArchTable=mod_dict[mod_id],vb=1)
    mod_dict[mod_id].add_bh(boreholes)
# Compute model
# Runs the model computation process for bhs, surfs, facies and props.
    of.process_model(mod_dict[mod_id])
    mod_folder = os.path.join(output_data, 'ArchPy_mods','ap_{}'.format(mod_dict[mod_id].name))  # Construct path as string
    #os.makedirs(mod_folder, exist_ok=True)  # Create directory if it doesn't exist
    mod_dict[mod_id].ws = mod_folder  # Assign the string path

    ap.inputs.save_project(mod_dict[mod_id])
    



    # Plotting stuff
    #mod_id='sm_1'
    por_facies1=mod_dict[mod_id].get_prop('Por')[0,0,0,:,0,:]
    comp_facies1=of.apply_porosity_compaction(mod_dict, mod_id, -0.0005).squeeze()
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
    im3 = axes[1,1].imshow(np.flipud(comp_facies1), cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axes[1,1].set_title("Compacted Porosity Field")
    axes[1,1].set_xlabel("column")
    
    # Add a shared colorbar
    cbar = fig.colorbar(im2, ax=axes[1,1], orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Porosity (-)")
    
    cbar = fig.colorbar(im1, ax=axes[0,1], orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Hydraulic Conductivity (m/day)")
    
    #Saving Figure
    fig.savefig('{}/figures/{}_summary.png'.format(output_data,mod_id), dpi=300, bbox_inches='tight')


#%% Computing compacted Porosity field
# mod_id='sm_10'
# por_facies1=mod_dict[mod_id].get_prop('Por')
# #get the 2D porosity array out of the default 5D array returned bz get_prop function.
# por_facies1=por_facies1[0,0,0,:,0,:]
# # Initialize a 2D array to store the compacted porosity
# comp_facies1 = np.full_like(por_facies1, por_facies1)  
# #Predefined porosity compaction coeff. 
# comp_coeff=-0.0005

# #make depth array
# z_vals=mod_dict[mod_id].get_zgc()

# ##Expanding the depth vector to a 2D array 
# z_vals=z_vals[:,np.newaxis]
# z_array = np.tile(z_vals, (1, por_facies1.shape[1]))
# mask=z_array<0 # Boolean mask for negative depth values

# #Applying the compaction to the porosity field
# comp_facies1[mask]=of.expo_trend(-z_array[mask],por_facies1[mask],comp_coeff)


# # Define a shared color scale
# # Define a shared color scale ignoring NaN values
# vmin = 0.1
# vmax = 0.65
# # Create subplots
# fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True, sharey=True)

# # Plot Porosity Field
# im1 = axes[0].imshow(np.flipud(por_facies1), cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
# axes[0].set_title("Original Porosity Field")

# axes[0].set_ylabel("i")

# # Plot Compacted Porosity Field
# im2 = axes[1].imshow(np.flipud(comp_facies1), cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
# axes[1].set_title("Compacted Porosity Field")
# axes[1].set_xlabel("k")

# # Add a shared colorbar
# cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
# cbar.set_label("Porosity")


#Saving Figure

