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
import re

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
    por_facies1 = por_facies1[0, 0, 0, :, :, :]

    # Initialize compacted porosity array with the original values
    comp_facies1 = np.copy(por_facies1)

    # Get depth values and expand into 2D array
    z_vals = mod_dict[mod_id].get_zgc()
    z_vals_3D = np.reshape(z_vals,(por_facies1.shape[0], 1,1))# Convert 1D to 3D
    z_array = np.broadcast_to(z_vals_3D, (por_facies1.shape[0], por_facies1.shape[1],por_facies1.shape[2]))  # Repeat along columns

    # Boolean mask for negative depth values
    #mask = z_array < 0

    # Apply compaction function only for negative depth values
    comp_facies1 = expo_trend(-z_array, por_facies1, comp_coeff)

    return comp_facies1

#****************************************************************#
# QC and Post processing
#****************************************************************#

def check_model_runs(base_folder, num_realizations):
    """
    Checks for successful MODFLOW model runs in a Monte Carlo simulation.

    Args:
        base_folder (str): The base directory containing the model subfolders.
        num_realizations (int): The total number of model realizations.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'model_name' and 'complete_run'
                          (1 for complete, 0 for incomplete), and the overall success rate.
    """

    results = []
    successful_runs = 0
    failed_runs = []
    complete_runs = []

    for i in range(1, num_realizations + 1):
        model_name = f'sm_{i}'
        model_folder = os.path.join(base_folder, model_name)
        output_file = os.path.join(model_folder, 'concvelo.tec')  # Corrected file name

        if os.path.exists(output_file):
            results.append({'model_name': model_name, 'complete_run': 1})
            complete_runs.append(model_name)
            successful_runs += 1
        else:
            results.append({'model_name': model_name, 'complete_run': 0})
            failed_runs.append(model_name)

    df = pd.DataFrame(results)
    success_rate = (successful_runs / num_realizations) * 100
    print(f"Success Rate: {success_rate:.2f}%")
    return df, complete_runs,failed_runs

def count_nper_infile(filename, search_string):
    '''counts how many stress periods are written out in the output file assuming the marker
    is known and input as search_string. eg. ZONE  T='''
    count = 0
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            count += line.count(search_string)
    return count

def create_results_folder(mod_dir):
    ''' creates the folder to store modified model output files for each stress period '''
    results_path = os.path.join(mod_dir, 'results')
    os.makedirs(results_path, exist_ok=True)
    return results_path

def split_sp_outputs(file_path,out_dir):
    '''A function to read the concvelo output file and split each stress period into a seperate 
    file. 
    inputs: path to convelo.tec file in model folder
            output folder to store the split files
    returns: indivual files for each stress period with the format results_sp{i}'''
    # Read the entire file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    zone_indices = []
    # Find all lines where 'ZONE T=' appears
    for i, line in enumerate(lines):
        if re.match(r'\s*ZONE\s+T\s*=', line):
            zone_indices.append(i)
    
    # Add end of file to help with last chunk
    zone_indices.append(len(lines))
    
    # Split and save each zone section
    for i in range(len(zone_indices) - 1):
        start_idx = zone_indices[i]
        end_idx = zone_indices[i + 1]
        chunk_lines = lines[start_idx:end_idx]
    
        # Include headers if needed — find the first "VARIABLES=" above the first zone
        if i == 0:
            header_lines = []
            for j in range(start_idx - 1, -1, -1):
                if "VARIABLES=" in lines[j]:
                    header_lines = lines[j:start_idx]
                    break
            chunk_lines = header_lines + chunk_lines
    
        # Save to file
        output_file = os.path.join(out_dir, f"results_sp{i + 1}.tec")
        with open(output_file, 'w') as out_file:
            out_file.writelines(chunk_lines)
    
        print(f"Saved: {output_file}")
        
def clean_tec_file(input_file, search_string='ZONE T='):
    # Create output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cleaned{ext}"

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if search_string in line:
                continue  # skip lines with 'ZONE T='
            elif 'VARIABLES=' in line:
                cleaned_line = line.replace('VARIABLES=', '').lstrip()
                outfile.write(cleaned_line)
            else:
                outfile.write(line)

    return output_file

# def clean_tec_file2(input_file, search_string='ZONE T='):
#     """
#     This function cleans a .tec file by removing header lines, 'VARIABLES=' line,
#     and the last lines starting with 'TEXT' and the line after.

#     Args:
#         input_file: The path to the .tec file.
#         search_string: The string to search for to skip lines.
#     Returns:
#         The name of the cleaned output file.
#     """
#     # Create output filename
#     base, ext = os.path.splitext(input_file)
#     output_file = f"{base}_cleaned{ext}"

#     with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
#             open(output_file, 'w', encoding='utf-8') as outfile:
#         lines = infile.readlines()  # Read all lines into a list
        
#         # Process lines, skipping unwanted ones
#         cleaned_lines = []
#         i = 0
#         while i < len(lines):
#             line = lines[i]
#             if search_string not in line:
#                 if 'VARIABLES=' in line:
#                     cleaned_line = line.replace('VARIABLES=', '').lstrip()
#                     cleaned_lines.append(cleaned_line)
#                 else:
#                     cleaned_lines.append(line)
#             i += 1

#         # Remove the last lines starting with "TEXT" and the line after
#         while cleaned_lines and cleaned_lines[-1].startswith("TEXT"):
#             cleaned_lines.pop()
            
#         # Write the cleaned lines to the output file
#         outfile.writelines(cleaned_lines)

#     return output_file



def clean_tec_file2(input_file, search_string='ZONE T='):
    """
    This function cleans a .tec file by removing header lines, the entire line
    containing 'VARIABLES=', and the last lines starting with 'TEXT' and the line after.
    It also explicitly removes the specific header line:
    "X", "Y", "Z" , "HEAD" , "CONC" , "VX" , "VY" , "VZ"

    Args:
        input_file: The path to the .tec file.
        search_string: The string to search for to skip lines.
    Returns:
        The name of the cleaned output file.
    """
    # Create output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cleaned{ext}"

    # Define the specific header to remove
    problematic_header = '"X", "Y", "Z" , "HEAD" , "CONC" , "VX" , "VY" , "VZ"'

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()  # Read all lines into a list
        
        # Process lines, skipping unwanted ones
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip() # Use strip to remove leading/trailing whitespace for comparison

            # Skip the line if it contains 'VARIABLES='
            if 'VARIABLES=' in line:
                i += 1
                continue # Move to the next line immediately

            # Skip the line if it's the specific problematic header
            if line == problematic_header:
                i += 1
                continue # Move to the next line immediately

            # Skip the line if it contains the search_string (e.g., 'ZONE T=')
            if search_string in line:
                i += 1
                continue # Move to the next line immediately

            # If none of the above conditions met, add the line (with its original newline character)
            cleaned_lines.append(lines[i]) # Append the original line from 'lines' list

            i += 1

        # Remove the last lines starting with "TEXT" and the line after
        while cleaned_lines and cleaned_lines[-1].strip().startswith("TEXT"):
            cleaned_lines.pop()
            
        # Write the cleaned lines to the output file
        outfile.writelines(cleaned_lines)

    return output_file


def print_last_lines(filename, num_lines=10):
    """
    Prints the last N lines of a file.

    Args:
        filename: The path to the file.
        num_lines: The number of lines to print (default: 10).
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return  # Exit the function if the file doesn't exist

    try:
        with open(filename, "r") as f:
            lines = f.readlines()  # Read all lines into a list
            if len(lines) >= num_lines:
                last_lines = lines[-num_lines:]  # Get the last N lines
            else:
                last_lines = lines  # If the file has fewer than N lines, get all of them
            for line in last_lines:
                print(line.strip())  # Print each line without leading/trailing whitespace
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def split_file_by_stress_period(input_filepath):
    """
    Splits a text file into multiple files based on lines containing
    '# stress period n'. Each new file is named 'ssm_sp_n.txt' and
    is written to the same directory as the input file.

    Args:
        input_filepath (str): The path to the input text file.
    """
    # Ensure the input file exists
    if not os.path.exists(input_filepath):
        print(f"Error: Input file '{input_filepath}' not found.")
        return

    # Get the directory of the input file
    input_directory = os.path.dirname(input_filepath)

    current_output_file = None
    stress_period_number = None

    try:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                # Check if the line indicates a new stress period
                match = re.search(r'# stress period (\d+)', line)
                if match:
                    # Extract the stress period number
                    new_stress_period_number = int(match.group(1))

                    # If there was an open file, close it before opening a new one
                    if current_output_file:
                        current_output_file.close()
                        # Use os.path.basename to get just the filename for printing
                        print(f"Closed: {os.path.basename(current_output_file.name)}")

                    # Update the stress period number
                    stress_period_number = new_stress_period_number

                    # Define the new output filename, joining it with the input directory
                    output_filename = os.path.join(input_directory, f"ssm_sp_{stress_period_number}.txt")

                    # Open the new output file in write mode
                    current_output_file = open(output_filename, 'w')
                    print(f"Opened: {os.path.basename(output_filename)}") # Print only the filename for clarity

                # Write the current line to the open output file, if any
                if current_output_file:
                    current_output_file.write(line)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the last opened file is closed
        if current_output_file:
            current_output_file.close()
            print(f"Closed: {os.path.basename(current_output_file.name)}")
            
def find_first_active(arr):
    """
    Returns the row index of the first occurrence of '1' in each column,
    searching from the top (row 0). If '1' is not found in a column, -1 is returned for that column.

    Args:
        arr (np.ndarray or list of lists): A 2D array (or list of lists) where you want to find the indices.

    Returns:
        np.ndarray: A 1D array where each element is the row index of the first '1'
                    for the corresponding column, or -1 if '1' is not found.
    """
    arr = np.array(arr) # Ensure it's a NumPy array

    # 1. Create a boolean array where True indicates a '1'
    is_one = (arr == 1)

    # 2. Use np.argmax along axis=0 (columns) to find the first True index.
    #    Important Note: If no True is found, np.argmax returns 0.
    first_one_indices = np.argmax(is_one, axis=0)

    # 3. Determine which columns actually contain at least one '1'.
    #    This helps us differentiate a '1' at index 0 from no '1' found.
    column_has_one = np.any(is_one, axis=0)

    # 4. Use np.where to set -1 for columns where '1' was not found,
    #    otherwise use the index found by argmax.
    result = np.where(column_has_one, first_one_indices, -1)

    return result


def find_last_col(arr, value):
    """
    Finds the index of the last column in a 2D NumPy array that contains a specific value.

    Args:
        arr (np.ndarray): The 2D input array.
        value (int or float): The value to search for.

    Returns:
        int: The index of the last column containing the value, or -1 if not found.
    """
    # 1. Check which columns contain the value (at least once)
    # axis=0 operates column-wise. np.any checks if any value is True in that column.
    cols_with_value = np.any(arr == value, axis=0)
    
    # 2. Find the indices of all columns that contain the value
    indices = np.where(cols_with_value == True)
    
    # 3. If any such columns were found, return the last index
    if indices[1].size > 0:
        return indices[1][-1]
    else:
        return -1 # Return -1 if the value was not found in any column
    
def calc_eq_fw_head(rho_s, rho_f, h_s, z):
    """
    Calculates the equivalent freshwater head.

    This function is used in variable-density groundwater modeling (e.g., with SEAWAT)
    to convert a saline water head to a standardized freshwater head, accounting for
    the density difference and the elevation of the measurement point.

    Args:
        rho_s (float): The density of the saline water (e.g., seawater in kg/m^3).
        rho_f (float): The density of freshwater (reference density, typically 1000 kg/m^3).
        h_s (float): The elevation of the saline water head (e.g., sea level elevation in meters).
        z (float): The elevation of the point of measurement (e.g., the model cell center in meters).

    Returns:
        float: The calculated equivalent freshwater head (h_f) in meters.
    """
    if rho_f <= 0:
        raise ValueError("Freshwater density (rho_f) must be positive.")

    # Formula to calculate equivalent freshwater head
    h_f = (rho_s / rho_f) * h_s - ((rho_s - rho_f) / rho_f) * z
    
    return h_f