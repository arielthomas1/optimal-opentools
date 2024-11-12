# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:07:34 2024

@author: Ariel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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