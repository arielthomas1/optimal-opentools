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

import numpy as np
import seaborn as sns
import pandas as pd
import optimal_functions as of
import matplotlib.pyplot as plt
import math
import geone
import geone.covModel as gcm
import os
import sys
import pyvista as pv
import ArchPy as ap
from scipy.stats import norm
from scipy import stats
# Set the seed
np.random.seed(777)

#intializing parameter propery table
# List of parameters
parameters = ['sw', 'sbd','cust', 'cst', 'tst', 'te_10km', 'tos']
mean_vals = [110, 130, 300,150, 100, 700, 1500 ]
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
df_cst_cleaned['ate_to_sed'].mean()

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


#%% Building bounding surfaces for the surrogate model 
''' MOdel top and base will be determined by extracting a random value for each parameter
extracted from the distributions visualized in the previous step'''

#TODO Create function to make base of surrogate model using random values of inputs
#TODO Create function to make top of surrogate model using random values from input parameter space 
#TODO Develop a methodology to QC the shape of the model create and cast out unrealistic creations

mod_id='sm_1'

#Model dimensions

# INLAND EXTENT
inland_sect=10000

# TOP ELEVATION
#retrieving random val based on the mean and std dev from the parameters table
top_elev=np.round(norm.rvs(loc=of.get_mean('te_10km',par_stats), scale=of.get_sdev('te_10km',par_stats)),-1)

#TODO - check and add skewness from real data

# TERRESTRIAL SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
tst=np.round(norm.rvs(loc=of.get_mean('tst',par_stats), scale=of.get_sdev('tst',par_stats)),-1)

# COAST SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
cst=np.round(norm.rvs(loc=of.get_mean('cst',par_stats), scale=of.get_sdev('cust',par_stats)),-1)
cust=np.round(norm.rvs(loc=of.get_mean('cst',par_stats), scale=of.get_sdev('cust',par_stats)),-1)

#SHELF WIDTH
#retrieving random val based on the mean and std dev from the parameters table
sw=np.round(norm.rvs(loc=of.get_mean('sw',par_stats), scale=of.get_sdev('sw',par_stats)),-1)
sw*=1000 #convert to SI units (m)

#SHELF EDGE SEDIMENT THICKNESS
#retrieving random val based on the mean and std dev from the parameters table
sbd=np.round(norm.rvs(loc=of.get_mean('sbd',par_stats), scale=of.get_sdev('sbd',par_stats)),-1)


#TOE OF SLOPE
#retrieving random val based on the mean and std dev from the parameters table

tos=np.round(norm.rvs(loc=of.get_mean('tos',par_stats), scale=of.get_sdev('tos',par_stats)),-1)


slope_width=20000 # hor. distance between shelf break and toe of slope in m
mod_len=inland_sect+int(sw)+slope_width # Defining the simulation grid size. 

# Calculating Average slope angle in degrees
'''>The inverse tangent of the depth of shelf break/shelf width '''
slope_angle = np.round(math.degrees(math.atan(sbd/sw)),2)

#Creating synthentic Boreholes to Anchor surrogate model surfaces.
'''Four boreholes will be created for each SM realization. 
The boreholes a, b, c and d will represent the control points at the
 inland mark, coast, shelf break and toe of slope, respectively.'''

with open(r"{}\surrogate_boreholes\bh_{}.lbh".format(output_data,mod_id),"w") as file:
    file.write('bh_id,bh_x,bh_y,bh_z,bh_td \n')
    file.write('a,0,0,{},{}\n'.format(top_elev,tst+100)) # inland anchor point
    file.write('b,{},0,0,{}\n'.format(inland_sect,cst+100)) # coastal anchor point
    file.write('c,{},0,{},{}\n'.format(inland_sect+sw,sbd,(2.5*cst)+100)) # shelf anchor point
    file.write('d,{},0,{},{}\n'.format(inland_sect+sw+slope_width,tos,tos+100)) # shelf anchor point

#Generating list of unit data in boreholes


#MODEL TOP
# MODEL BASE

print('Model parameter summary (m): \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total Model length = {} \n Top Elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}'.format(sw,slope_width,sbd,slope_angle,inland_sect,mod_len,top_elev,tst,cst))

with open(r"{}\surrogate_mod_summary\{}.txt".format(output_data,mod_id), "w") as file:
    file.write('Model parameter summary (m): \n Shelf width = {} \n Slope width = {} \n Shelf break depth = {} \n Shelf gradient (deg)= {} \n Inland length = {} \n Total Model length = {} \n Top Elevation = {} \n Terrestrial sediment = {}  \n Coastal sediment = {}'.format(sw,slope_width,sbd,slope_angle,inland_sect,mod_len,top_elev,tst,cst))
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
nz=int(top_elev-tos/sz)
ox=oy=oz=0 
#oz=top_elev #
dimensions= (nx,ny,nz)
spacing = (sx, sy, sz)
origin = (ox, oy, oz)
#adding grid
sm_1.add_grid(dimensions, spacing, origin) 

#Stratigraphic Pile
strat_sm_1=ap.base.Pile('Strat')
sm_1.set_Pile_master(strat_sm_1)

#defining top and base surface objects
top=ap.base.Surface(name='top_mod',
                    dic_surf={'N_transfo': False, 'covmodel': None, 'int_method': 'kriging'},
                    contact='onlap')

base=ap.base.Surface(name='base_mod',
                     dic_surf={'N_transfo': False, 'covmodel': None, 'int_method': 'kriging'},
                     contact='onlap')

#Unit 1 - sedimentary unit
#NO covmodel defined at this stage because the model domain is being expicitly defined as homogeneous.
u1=ap.base.Unit(name= 'sed_lyr',
                order=1,
                color='brown',
                surface=top)
u2=ap.base.Unit(name='base_lyr',
                order=2,
                color='grey',
                surface=base)

#Adding unit to the strat pile
strat_sm_1.add_unit([u1,u2])

#Adding Facies
sand=ap.base.Facies(ID=1, name='Sand', color='yellow')
silt= ap.base.Facies(ID=2, name='silt',color='darkorange')
clay=ap.base.Facies(ID=3, name='clay',color='brown')
gravel=ap.base.Facies(ID=4, name='gravel',color='palegreen')
rock=ap.base.Facies(ID=5, name='rock', color='darkgray' )


u1.add_facies([silt])
u2.add_facies([rock])

#Adding properties
'''Porosity typically exhibits a gaussian normal distribution so this type
of model will be assigned.'''

cov_mod_por =gcm.CovModel2D(elem=[('gaussian',{'w':1.0,'r':[1000,100]})],
                             alpha=-slope_angle)

mean_vals=[0.35,0.5]
list_facies=[silt,rock]

por=ap.base.Prop("Por",facies=list_facies,
                 covmodels=cov_mod_por,
                 means=mean_vals,
                 int_method="sgs")

#Adding porosity to model object'
sm_1.add_prop(por)



