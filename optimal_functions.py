# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:07:34 2024

@author: Ariel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import norm

def plot_gaussian_distribution(variable_name, mean, std_dev):
    # Generate values for the x-axis
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100000)
    
    # Gaussian distribution formula
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
    
    # Plot the distribution
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='blue', label=f'{variable_name}')
    
    # Set title and labels
    plt.title(f'{variable_name}|$\mu$ - {mean}| $\sigma$ - {std_dev}')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    # Display the plot
    plt.grid(True)
    plt.show()
    
def clean_null_values(df, column_name):
    """
    Removes rows with null values in a specified column of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    column_name (str): The name of the column to check for null values.
    
    Returns:
    pd.DataFrame: A new DataFrame with null values removed from the specified column.
    """
    # Drop rows where the specified column has null values
    df_cleaned = df.dropna(subset=[column_name])
    
    return df_cleaned
# Example usage
def get_mean(par_name,df):
    '''a function to retrieve the mean value from the table of parameter values based on the input of the parameter acronym'''
    stats =df.loc[df['parameter']=='{}'.format(par_name),['mean', 'std_dev']].values[0][0:2]
    return stats[0]

def get_sdev(par_name,df):
    '''a function to retrieve the mean value from the table of parameter values based on the input of the parameter acronym'''
    stats =df.loc[df['parameter']=='{}'.format(par_name),['mean', 'std_dev']].values[0][0:2]
    return stats[1]

def get_random_top_elev(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for top elevation based on mean and standard deviation 
    from the parameters table. Ensures the value is >= 1.

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'te_10km').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: A valid top elevation value rounded to the nearest 10.
    """
    top_elev = -1  # Initialize with a negative value to enter the loop
    while top_elev < 1:
        top_elev = np.round(norm.rvs(
            loc=mean_func(param_name, par_stats),
            scale=sdev_func(param_name, par_stats),
            random_state=rng.integers(10000)
        ), -1)
    return top_elev

def get_random_cst(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for coastal sediment thickness (cst) based on the mean 
    and standard deviation from the parameters table. Ensures the value is >= 0.

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'cst').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: A valid coastal sediment thickness value rounded to the nearest 10.
    """
    cst = -1  # Initialize with a negative value to enter the loop
    while cst < 0:
        cst = np.round(
            norm.rvs(
                loc=mean_func(param_name, par_stats),
                scale=sdev_func(param_name, par_stats),
                random_state=rng.integers(10000)
            ), -1
        )
    return cst

def get_random_cust(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for the 'cust' parameter based on the mean 
    and standard deviation from the parameters table. Ensures the value is >= 0.

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'cust').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: A valid 'cust' parameter value rounded to the nearest 10.
    """
    cust = -1  # Initialize with a negative value to enter the loop
    while cust < 0:
        cust = np.round(
            norm.rvs(
                loc=mean_func(param_name, par_stats),
                scale=sdev_func(param_name, par_stats),
                random_state=rng.integers(10000)
            ), -1
        )
    return cust

def get_shelf_width(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for the shelf width (sw) parameter based on the mean 
    and standard deviation from the parameters table, and convert it to SI units (meters).

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'sw').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: Shelf width value in meters, rounded to the nearest 10.
    """
    sw = np.round(
        norm.rvs(
            loc=mean_func(param_name, par_stats),
            scale=sdev_func(param_name, par_stats),
            random_state=rng.integers(10000)
        ), -1
    )
    return sw * 1000  # Convert to SI units (meters)

def get_shelf_edge_thickness(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for the shelf edge sediment thickness (sbd) parameter 
    based on the mean and standard deviation from the parameters table.

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'sbd').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: Shelf edge sediment thickness value, rounded to the nearest 10.
    """
    sbd = np.round(
        norm.rvs(
            loc=mean_func(param_name, par_stats),
            scale=sdev_func(param_name, par_stats),
            random_state=rng.integers(10000)
        ), -1
    )
    return sbd

def get_tos_value(mean_func, sdev_func, param_name, par_stats, rng):
    """
    Retrieve a random value for the 'tos' parameter based on the mean 
    and standard deviation from the parameters table.

    Args:
        mean_func (function): Function to get the mean of the parameter.
        sdev_func (function): Function to get the standard deviation of the parameter.
        param_name (str): Name of the parameter (e.g., 'tos').
        par_stats (object): Parameter statistics table.
        rng (np.random.Generator): Random number generator.

    Returns:
        float: The 'tos' value, rounded to the nearest 10.
    """
    tos = np.round(
        norm.rvs(
            loc=mean_func(param_name, par_stats),
            scale=sdev_func(param_name, par_stats),
            random_state=rng.integers(10000)
        ), -1
    )
    return tos

def calculate_slope_angle(sbd, sw):
    """
    Calculate the slope angle in degrees based on the shelf break depth and shelf width.
    
    Parameters:
    sbd (float): Depth of the shelf break
    sw (float): Shelf width
    
    Returns:
    float: Slope angle in degrees, rounded to two decimal places
    """
    slope_angle = np.round(math.degrees(math.atan(sbd / sw)), 2)
    return slope_angle

def extract_mod_parameter(file_path, parameter_name):
    """
    Extract a specific numerical value from the model parameter summary file.

    Parameters:
    file_path (str): The path to the file.
    parameter_name (str): The name of the parameter to extract (e.g., "Top Elevation").

    Returns:
    float or None: The extracted value, or None if the parameter is not found.
    """
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and check if the parameter name is in the line
            line = line.strip()
            if parameter_name in line:
                # Extract and return the numerical value after "="
                return float(line.split('=')[-1].strip())
    
    # Return None if the parameter was not found
    print(f"Parameter '{parameter_name}' not found in file.")
    return None

def read_mod_file(folder_path, mod_id, parameter_name):
    """
    Reads a specific file and extracts the requested parameter value.

    Parameters:
    folder_path (str): Path to the folder containing the files.
    file_index (int): Model realization number to determine the file name.
    parameter_name (str): The parameter to extract.

    Returns:
    float or None: The extracted parameter value.
    """
    file_name = f"{mod_id}.txt"  # Adjust file extension if necessary
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        return extract_mod_parameter(file_path, parameter_name)
    else:
        print(f"File {file_name} not found.")
        return None
    
def process_model(model_obj):
    """Runs the required processing functions on the given ArchPy model object."""
    model_obj.process_bhs()
    model_obj.compute_surf(1)
    model_obj.compute_facies(1)
    model_obj.compute_prop(1)

def expo_trend(z, a, b):
    '''Calculates the compacted porosity value at  a depth z (m) based on the initial porosity
    a and the compaction coeff b
    
    Parameters:
        a: initial porosity - i.e. at surface
        b: compaction coeff
        z: depth below surface in m '''
    
    return a*np.exp(b*z)


def apply_porosity_compaction(mod_dict, mod_id, comp_coeff):
    """
    Extracts the porosity field from a given model in mod_dict and applies a compaction function.

    Parameters:
    mod_dict (dict): Dictionary storing surrogate models.
    mod_id (int or str): Key corresponding to the model in mod_dict.
    comp_coeff (float): Porosity compaction coefficient.

    Returns:
    np.ndarray: The compacted porosity field.
    """
    # Extract porosity field
    por_facies1 = mod_dict[mod_id].get_prop('Por')
    
    # Convert to 2D array (extract relevant slice from 5D array)
    por_facies1 = por_facies1[0, 0, 0, :, 0, :]

    # Initialize compacted porosity array with the original values
    comp_facies1 = np.copy(por_facies1)

    # Get depth values and expand into 2D array
    z_vals = mod_dict[mod_id].get_zgc()[:, np.newaxis]  # Convert 1D to 2D
    z_array = np.tile(z_vals, (1, por_facies1.shape[1]))  # Repeat along columns

    # Boolean mask for negative depth values
    mask = z_array < 0

    # Apply compaction function only for negative depth values
    comp_facies1[mask] = expo_trend(-z_array[mask], por_facies1[mask], comp_coeff)

    return comp_facies1
