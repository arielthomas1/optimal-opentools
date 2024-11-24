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
work_dir=r"H:\My Drive\OPTIMAL\Project work\optimal"
os.chdir(work_dir)
import numpy as np
import seaborn as sns
import pandas as pd
import optimal_functions as of
import matplotlib.pyplot as plt
import math
import random
import geone
import geone.covModel as gcm

import sys
import pyvista as pv
import ArchPy as ap
from scipy.stats import norm
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator
# Set the seed
np.random.seed(777)

#intializing parameter propery table
# List of parameters
parameters = ['sw', 'sbd','cust', 'cst', 'tst', 'te_10km', 'tos']
mean_vals = [110, 130, 300,150, 100, 700, 3000 ]
std_val = [30, 20, 100,10, 50, 150, 500 ]
skew = [3,0,0,0,0,0,0]
kurtosis = [8,3,3,3,3,3,3]
par_stats = pd.DataFrame({
    'parameter': parameters,
    'mean': mean_vals,
    'std_dev': std_val,
    'skew': skew,
    'kurt': kurtosis
})


input_data=r'H:\My Drive\OPTIMAL\Project work\optimal\surrogate_mode_tables'
output_data=r"H:\My Drive\OPTIMAL\Project work\optimal\surrogate_sections"

#%% Visualizing the distribution of surrogate model input parameters

# Create a figure and axes
fig1=plt.figure(figsize=(10, 6))

# Initialize the x-axis range to accommodate all distributions
min_value = float('inf')
max_value = float('-inf')

# Loop through each row of the DataFrame to calculate distributions
for index, row in par_stats.iterrows():
    parameter = row['parameter']
    mean = row['mean']
    std_dev = row['std_dev']
    
    # Generate data for the distribution based on mean and std_dev extracted from global datasets
    distribution_data = np.random.normal(loc=mean, scale=std_dev, size=10000)
    
    # Plot the KDE on the same axes, including the mean and std. dev. of each parameter in the legend
    sns.kdeplot(distribution_data, fill=True, label=f'{parameter} - $\mu$: {mean} , $\sigma$: {std_dev}')
    
    # Adjust the range for the x-axis to fit all distributions
    min_value = min(min_value, mean - 4 * std_dev)
    max_value = max(max_value, mean + 4 * std_dev)

# Set the x-axis limits based on the largest distribution
plt.xlim(min_value, max_value)

# Add title and legend
plt.title('KDE of Surrogate model input parameters')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Show plot
plt.show()
fig1.savefig(r'H:\My Drive\OPTIMAL\Project work\Figures\KDE_input_params.png', dpi=450, bbox_inches='tight')


#%% Coastal Sediment Thickness

# Importing coastal sediment thickness estimate dataset from csv table
df_cst_zam=pd.read_csv('{}\coastal_unconsol_thickness_zamrsky.csv'.format(input_data),delimiter=';')

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
df_10km=pd.read_csv(r'{}\Australia_10KM_Buffer_Elevation.csv'.format(input_data),delimiter=';')
df_10km_clean=of.clean_null_values(df_10km, 'Australia_elevation')
# Data from 20 KM Buffer
df_20km=pd.read_csv(r'{}\Australia_20KM_Buffer_Elevation.csv'.format(input_data),delimiter=';')
df_20km_clean=of.clean_null_values(df_20km, 'Australia_elevation')
# Data from 30 KM Buffer
df_30km=pd.read_csv(r'{}\Australia_30KM_Buffer_Elevation.csv'.format(input_data),delimiter=';')
df_30km_clean=of.clean_null_values(df_30km, 'Australia_elevation')

# writing stats to the master parameter dataframe
par_stats.loc[par_stats['parameter']=='te_10km','mean']=df_10km_clean['Australia_elevation'].mean().round(1)
par_stats.loc[par_stats['parameter']=='te_10km','std_dev']=df_10km_clean['Australia_elevation'].std().round(1)
par_stats.loc[par_stats['parameter']=='te_10km','skew']=df_10km_clean['Australia_elevation'].skew().round(1)
par_stats.loc[par_stats['parameter']=='te_10km','kurt']=df_10km_clean['Australia_elevation'].kurt().round(1)

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
# Define the random seed
SEED = np.random.randint(239)
rng = np.random.default_rng(SEED)
mod_id='sm_1'

#Model dimensions

# INLAND EXTENT
inland_sect=10000

# TOP ELEVATION
#retrieving random val based on the mean and std dev from the parameters table
top_elev = -1  # Initialize with a negative value to enter the loop
while top_elev <1:
    top_elev=np.round(norm.rvs(
                    loc=of.get_mean('te_10km',par_stats), 
                    scale=of.get_sdev('te_10km',par_stats),
                    random_state=rng.integers(10000)),-1)

#TODO - check and add skewness from real data

# TERRESTRIAL SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
tst=np.round(norm.rvs(
                loc=of.get_mean('tst',par_stats), 
                scale=of.get_sdev('tst',par_stats),
                random_state=rng.integers(10000)),-1)

# COAST SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
# Draw a value from the distribution, retry if it's negative
#GLOBSED
cst = -1  # Initialize with a negative value to enter the loop
while cst < 0:
    cst = np.round(
        norm.rvs(
            loc=of.get_mean('cst', par_stats),
            scale=of.get_sdev('cst', par_stats),
            random_state=rng.integers(10000)),-1)
#cust=np.round(norm.rvs(loc=of.get_mean('cust',par_stats), scale=of.get_sdev('cust',par_stats)),-1)
# Draw a value from the distribution, retry if it's negative
#ATE - Zamrsky
cust = -1  # Initialize with a negative value to enter the loop
while cust < 0:
    cust = np.round(
        norm.rvs(
            loc=of.get_mean('cust', par_stats),
            scale=of.get_sdev('cust', par_stats),
            random_state=rng.integers(10000)),-1)
#SHELF WIDTH
#retrieving random val based on the mean and std dev from the parameters table
sw=np.round(norm.rvs(
            loc=of.get_mean('sw',par_stats), 
            scale=of.get_sdev('sw',par_stats),
            random_state=rng.integers(10000)),-1)
sw*=1000 #convert to SI units (m)

#SHELF EDGE SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
sbd=np.round(norm.rvs(
            loc=of.get_mean('sbd',par_stats), 
            scale=of.get_sdev('sbd',par_stats),
            random_state=rng.integers(10000)),-1)

#TOE OF SLOPE
#retrieving random val based on the mean and std dev from the parameters table

tos=np.round(norm.rvs(
            loc=of.get_mean('tos',par_stats), 
            scale=of.get_sdev('tos',par_stats),
            random_state=rng.integers(10000)),-1)


slope_width=20000 # hor. distance between shelf break and toe of slope in m
mod_len=inland_sect+int(sw)+slope_width # Defining the simulation grid size. 
unconsol_ratio=cust/cst
# Calculating Average slope angle in degrees
'''>The inverse tangent of the depth of shelf break/shelf width '''
slope_angle = np.round(math.degrees(math.atan(sbd/sw)),2)

# Finding the anchor point at the shelf break to form a straight base line
base_gradient=(-tos--cst)/(sw+slope_width)
intercept=-cst-base_gradient*0
base_angle = np.round(math.degrees(math.atan(base_gradient)))
z_sb_anchor=base_gradient*(sw)+intercept


shelf_anchor_base=np.minimum(tos,(2*cst))

#Defining anchor points
x_a=0 
x_b=int(inland_sect)
x_c=int(inland_sect+sw)
x_d=int(mod_len)
#Model top surface
z_a_top=top_elev
z_b_top=0
z_c_top=-sbd
z_d_top=-tos
#Model Base surface
z_a_base=top_elev-tst
z_b_base=0-cst
z_c_base=z_sb_anchor
z_d_base=-tos

x_mod=np.array([x_a,x_b,x_c,x_d])
z_top=np.array([z_a_top,z_b_top,z_c_top,z_d_top])
z_base=np.array([z_a_base,z_b_base,z_c_base,z_d_base])
#z_base_2=np.array([z_a_base,z_b_base,z_sb_anchor,z_d_base])

#Defining well total depths
td_a=top_elev+cst
td_b=cst +500
td_c=td_d=(tos-sbd)-5

#creating control wells at 5 km intervals
x_bhs=np.arange(0,sw,10000)
z_bhs=np.ones_like(x_bhs)
plt.plot(x_mod/1000, z_top,'-o', label='Top Model')      # Interpolated spline
plt.plot(x_mod/1000, z_base,'-o', label='Base Model')
plt.scatter(x_bhs/1000,z_bhs,'x',label='Well locations')
#plt.plot(x_mod/1000, z_base_2,'-x', label='Base Model')

plt.legend()
plt.xlabel("Distance (km)")
plt.ylabel("Depth (m)")
plt.title("Control Points QC Sanity Check")
plt.grid()
plt.show()
#%% Creating synthentic Boreholes to Anchor surrogate model surfaces.
'''Four boreholes will be created for each SM realization. 
The boreholes a, b, c and d will represent the control points at the
 inland mark, coast, shelf break and toe of slope, respectively.'''

# with open(r"{}\surrogate_boreholes\bh_{}.lbh".format(output_data,mod_id),"w") as file:
#     file.write('bh_ID,bh_x,bh_y,bh_z,bh_depth\n')
#     file.write('a,{},1,{},{}\n'.format(-inland_sect,top_elev,top_elev+tst)) # inland anchor point
#     file.write('b,{},1,0,{}\n'.format(-inland_sect,cst)) # coastal anchor point
#     file.write('c,{},1,-{},{}\n'.format(sw,sbd,(2*cst)-sbd)) # shelf anchor point
#     file.write('d,{},1,-{},{}\n'.format(sw+slope_width,tos,tos)) # slope anchor point
with open(r"{}\surrogate_boreholes\bh_{}.lbh".format(output_data,mod_id),"w") as file:
    file.write('bh_ID,bh_x,bh_y,bh_z,bh_depth\n')
    file.write('a,1,50,{},{}\n'.format(top_elev/10,td_a/10)) # inland anchor point
    file.write('b,{},50,5,{}\n'.format(inland_sect/100,td_b/10)) # coastal anchor point
    file.write('c,{},50,{},{}\n'.format((inland_sect+sw)/100,sbd/10,td_c/10)) # shelf anchor point
    #file.write('d,{},50,{},{}\n'.format((inland_sect+sw+slope_width-50)/100,sbd/10,td_d/10)) # slope anchor point
#Sanity checks
#TODO Add sanity checks for values to ensure they honour realistic geometry

#Generating list of unit data in boreholes
with open(r"{}\surrogate_boreholes\bh_{}.ud".format(output_data,mod_id),"w") as file:
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
with open(r"{}\surrogate_boreholes\bh_{}.fd".format(output_data,mod_id),"w") as file:
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
bh_folder=r"H:/My Drive/OPTIMAL/Project work/optimal/surrogate_sections/surrogate_boreholes"
bh_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.lbh'.format(mod_id)))
units_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.ud'.format(mod_id)))
facies_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.fd'.format(mod_id)))

#MODEL TOP

#MODEL BASE

print('Model parameter summary (m): \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total Model length = {} \n Top Elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}\n Unconsol. ratio = {}'.format(sw,slope_width,sbd,slope_angle,inland_sect,mod_len,top_elev,tst,cst,cust/cst))

with open(r"{}\surrogate_mod_summary\{}.txt".format(output_data,mod_id), "w") as file:
    file.write('Model parameter summary (m): \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total Model length = {} \n Top Elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}\n Unconsol. ratio = {}'.format(sw,slope_width,sbd,slope_angle,inland_sect,mod_len,top_elev,tst,cst,cust/cst))
file.close()

#%%
#CREATING ARCHPY MODEL OBJECT

sm_1=ap.base.Arch_table(mod_id,"{}".format(output_data),seed=777,verbose=1)
#TODO design loop for generating multiple mod_ids

'''The simulation grid will be designed such that all the modesl have the same cell size
despite having different dimensions. The number of cells (nx) will be a function 
of the model length (mod_len) i.e., nx=mod_len/100'''


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


#%%Creating interpolated Top and Base Surfaces 

#Defining the aquifer geometries using a fixed value relative to model bounds

# aq_len=random.randint(0, int(sw) // 100)*100
# ob_thickness=random.randint(0,300)
# aq1_thickness=random.randint(100,300)
aq_len=sw
ob_thickness=200
aq1_thickness=400

#Interpolating surface at model resolution
x_new = np.linspace(x_mod.min(), x_mod.max(), nx)
x_bhs=np.linspace(sx,sw,int(nx/100))

# Create a cubic B-spline 
spline_top = PchipInterpolator(x_mod, z_top)
z_top_new = spline_top(x_new)

spline_base = PchipInterpolator(x_mod, z_base)  
z_base_new = spline_base(x_new)

#Creating boreholes
z_bhs=spline_top(x_bhs)
z_unit_base=spline_base(x_bhs)
#setting the the total depth of the boreholes equal to the model domain
td_bhs=abs(spline_base(x_bhs)-z_bhs)
top_aq1=spline_top(x_bhs)-ob_thickness
bot_aq1=spline_top(x_bhs)-(ob_thickness+aq1_thickness)

lbh_header = ['bh_ID', 'bh_x', 'bh_y', 'bh_z', 'bh_depth']
fd_header=['bh_ID','facies_ID','top','bot']
ud_header=['bh_ID','Strat','top','bot']

df_lbh=pd.DataFrame(columns=lbh_header)
df_fd=pd.DataFrame(columns=fd_header)
df_ud=pd.DataFrame(columns=ud_header)
# Create the list of borehole names
borehole_names = [f"bh_{i}" for i in range(1, len(x_bhs) + 1)]
#Fill boreholes
df_lbh['bh_ID']=borehole_names
df_lbh['bh_x']=np.round(x_bhs/100)
df_lbh['bh_y']=(sy/2)*np.ones_like(x_bhs)
df_lbh['bh_z']=np.floor(z_bhs)
df_lbh['bh_depth']=np.floor(td_bhs)
df_lbh.to_csv(r"{}\surrogate_boreholes\bh_{}.lbh".format(output_data,mod_id),index=False)
#Fill Unit - U
# Loop to populate the DataFrame
for i, (bh_id, u_top, u_bot) in enumerate(zip(borehole_names, z_bhs, z_unit_base)):
    df_ud.loc[i] = [bh_id, 'U', np.floor(u_top),np.floor(u_bot)]  # Add values row by row


df_ud.to_csv(r"{}\surrogate_boreholes\bh_{}.ud".format(output_data,mod_id),index=False)
df_ud


#Fill facies
# Loop to populate the DataFrame
for i, (bh_id, f_top, f_bot) in enumerate(zip(borehole_names, top_aq1, bot_aq1)):
    df_fd.loc[i] = [bh_id, 'sand', np.floor(f_top),np.floor(f_bot)]  # Add values row by row
    
df_fd
df_fd.to_csv(r"{}\surrogate_boreholes\bh_{}.fd".format(output_data,mod_id),index=False)


#%%
#
# Plotting for QC
fig, ax = plt.subplots()

# Fill the area between the two lines
ax.fill_between(x_new/1000, z_top_new, z_base_new, color='orange', alpha=0.7)

plt.scatter(x_bhs/1000,z_bhs,marker='s',s=10,color='black',label='Boreholes')
plt.scatter(x_bhs/1000,-td_bhs,marker='s',s=10,color='black',label='Boreholes TD')
#Markers
plt.scatter(x_mod/1000, z_top, marker='x', color='gray',label='Control points')  # Original control points
plt.scatter(x_mod/1000, z_base, marker='x', color='gray')  
#Surfaces
plt.plot(x_new/1000, z_top_new, color='blue',label='Top model int')      # Interpolated spline
plt.plot(x_new/1000, z_base_new, color='black',label='Base model int')      # Interpolated spline
plt.plot(x_bhs/1000,top_aq1,color='yellow',label= 'Aquifer 1')
plt.plot(x_bhs/1000,bot_aq1,color='yellow')
ax.fill_between(x_bhs/1000, top_aq1, bot_aq1, color='yellow', alpha=0.7)

plt.legend()
plt.xlabel("Distance (km)")
plt.ylabel("Depth (m)")
plt.title("{} - Interpolation - QC ".format(mod_id))
plt.show()

fig.savefig('{}/figures/{}_geometry.png'.format(output_data,mod_id), dpi=450, bbox_inches='tight')
#%%
#3D Quick Viewwer
# #A vector to add a dimension to the 2D model
# y_vect=np.linspace(0,5,5)
# X_grid,Y_grid=np.meshgrid(x_new,y_vect)
# top_surf=np.array([z_top_new])
# bot_surf=np.array([z_base_new])
# # Plot the top and bottom surfaces
# fig = plt.figure(figsize=(10, 6))

# # Create a 3D axis
# ax = fig.add_subplot(111, projection='3d')
# # Plot the top surface
# ax.plot_surface(X_grid, Y_grid, top_surf , cmap='viridis', alpha=0.8, edgecolor='black', label="Top Surface")

# # Plot the bottom surface
# ax.plot_surface(X_grid, Y_grid, bot_surf, cmap='plasma', alpha=0.8, edgecolor='red', label="Bottom Surface")

# # Set axis labels
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Elevation')

# # Add a title
# ax.set_title('Top and Bottom Surfaces')

# # Show the plot
# plt.show()


#
# Creating top and base model arrays for input into model
top_surf=np.array([z_top_new])
bot_surf=np.array([z_base_new])
#adding grid
sm_1.add_grid(dimensions, spacing, origin, top=top_surf, bot=bot_surf) 

#Stratigraphic Pile
strat_pile=ap.base.Pile('Strat')
sm_1.set_Pile_master(strat_pile)
covmodel_er = gcm.CovModel2D(elem=[('spherical', {'w':2, 'r':[100,100]})])

#defining top and base surface objects
top=ap.base.Surface(name='top_mod',
                     dic_surf={'N_transfo': False, 'covmodel': covmodel_er, 'int_method': 'kriging'},
                     contact='erode')

#base=ap.base.Surface(name='base_mod',
 #                     dic_surf={'N_transfo': False, 'covmodel': covmodel_er, 'int_method': 'kriging'},
  #                    contact='onlap')

#Unit 1 - sedimentary unit
#NO covmodel defined at this stage because the model domain is being expicitly defined as homogeneous.
covmod_U=gcm.CovModel3D(elem=[('spherical', {'w':2, 'r':[5000,100,100]})],beta=2*slope_angle)
U_facies={'f_method':'SIS','f_covmodel':covmod_U}
U=ap.base.Unit(name= 'sed_lyr',
                order=1,
                color='brown',
                surface=top,
                dic_facies=U_facies)
#L=ap.base.Unit(name='base_lyr',
 #               order=2,
  #              color='grey',
   #             surface=base)
   
#TODO Add subpiles to include the aquifer interval as an explicit unit in the model



#pile = StratPile(f'strat_{mod_id}')      
#Adding unit to the strat pile
strat_pile.add_unit([U])
#ADDING SURFACES TO THE STRAT PILE


#Adding Facies
sand=ap.base.Facies(ID=1, name='Sand', color='yellow')
silt= ap.base.Facies(ID=2, name='silt',color='darkorange')
clay=ap.base.Facies(ID=3, name='clay',color='brown')
gravel=ap.base.Facies(ID=4, name='gravel',color='palegreen')
rock=ap.base.Facies(ID=5, name='rock', color='darkgray' )


U.add_facies([silt])
U.add_facies([sand])

#Adding properties
'''Porosity typically exhibits a gaussian normal distribution so this type
of model will be assigned.'''

cov_mod_por=gcm.CovModel3D(elem=[('gaussian',{'w':slope_angle,'r':[1000,10,50]})],
                             alpha=0,beta=slope_angle,gamma=0)

mean_vals=[0.35,0.5]
list_facies=[sand,silt]

por=ap.base.Prop("Por",facies=list_facies,
                 covmodels=[cov_mod_por],
                 means=mean_vals,
                 int_method="sgs",
                 vmin=0.10,
                 vmax=0.65)

#Adding porosity to model object'
sm_1.add_prop(por)
#%%
#Reading  borehole files
bh_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.lbh'.format(mod_id)))
units_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.ud'.format(mod_id)))
facies_path=pd.read_csv(os.path.join(bh_folder,'bh_{}.fd'.format(mod_id)))

# Importing Borehole data
db,l_bhs=ap.inputs.load_bh_files(list_bhs=bh_path,
                               units_data=units_path,
                               facies_data=facies_path,
                               altitude=True)

db
#%%
#Extracting boreholes
boreholes=ap.inputs.extract_bhs(df=db, list_bhs=l_bhs,ArchTable=sm_1)
sm_1.add_bh(boreholes)
#%% Compute model
sm_1.process_bhs()
#%%
sm_1.compute_surf(1)
#%%
sm_1.compute_facies(1)
#%%
#sm_1.compute_prop(1)
#%% get results

vex=15

pg=pv.Plotter()
sm_1.plot_grid(v_ex=vex)
#%%

sm_1.plot_units(iu=0,v_ex=vex)
#%%
sm_1.plot_facies(iu=0,ifa=0,v_ex=vex)

#%%
sm_1.plot_prop(por.name,v_ex=vex)
#%%
sm_1.plot_bhs()