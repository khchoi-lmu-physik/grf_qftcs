
# @title Import packages and libraries

from scipy.special import hankel2
from scipy.stats import skew, kurtosis

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import glob
import os
import gc
import time as ti

# @title 1.2 Figures plottings

def multiple_run_std_mean(npy_data, total_steps, time_steps):
    """
    Computes running the mean and standard deviation across
    multiple simulatiosns.

    Args:
        npy_data (list): List of file paths to the .npy files containing
        simulation data for each run.
        total_steps (int): Total number of time steps in the simulation.
        time_steps (float): Time increment between each simulation step.

    Returns:
        tuple: Two NumPy arrays, one containing the mean values and the other
        containing the standard deviations at each time step across all runs.
    """

    mean_array = []
    std_array  = []

    # Iterate over each time step to compute mean and standard deviation
    for steps in range(total_steps):

        # Compute the time at the current step
        time = (steps+1)*time_steps

        data = []

        # Collect data from all runs at the current time step
        for data_name in npy_data:

            # Load the  data from the .npy file
            data_array = np.load(f'{data_name}')

            # Append to the data list by extracting the Gassian random field data
            data.append(data_array[steps, 1])

        # Compute the mean and standard deviation at the current time step
        mean_array.append( [time, np.mean(data)] )
        std_array.append ( [time, np.std( data)] )

    return np.array(mean_array ), np.array(std_array )


def plot_multiple_runs( data_prefix, ylabel, total_steps, time_steps, xlabel= r'$\eta/\eta_0$' ):

    """
    Plots the statistics of multiple simulation runs on the same graph, with a
    the running mean and standard deviation across all simulations.

    Args:
        data_prefix (str): Prefix used to identify relevant .npy files.
        ylabel (str): Label for the y-axis.
        total_steps (int): Total number of time steps to plot.
        time_steps (float): Time increment of each simulation.
        xlabel (str): Label for the x-axis. Defaults to r'$\eta/\eta_0$'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # slow but clean
    fig = plt.figure(figsize=(8, 6))

    # Find all .npy files matching the data prefix
    npy_data = glob.glob(f'{data_prefix}_*.npy')

    # Plot each run individually with different styles for the first run
    for data_name in npy_data:

        # Load the data from the .npy file
        data = np.load(f'{data_name}')

        # Extract the run ID from the file name
        run_id = int(data_name.split('_')[-1].split('.')[0])

        # Set the line width and color
        linewidth = 2.5 if run_id == 1 else 1.25
        color = 'red' if run_id == 1 else 'green'
        alpha = 0.7 if run_id == 1 else 0.35
        zorder = 2 if run_id == 1 else 1

        # Plot the data for the current run
        plt.plot(data[:total_steps, 0], data[:total_steps, 1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

    # Compute the running mean and standard deviation across all simulation runs
    running_mean , running_std = multiple_run_std_mean(npy_data, total_steps, time_steps) #extract mean and std over all multiple run.

    # Plot the running mean
    plt.plot(running_mean[:total_steps, 0], running_mean[:total_steps, 1], color='black', alpha=0.8, linewidth=2.5, zorder=3)

    # Fill the area between the running mean +- runnning standard deviation
    plt.fill_between(running_mean[:total_steps, 0],
                    running_mean[:total_steps, 1] + running_std[:total_steps, 1],
                    running_mean[:total_steps, 1] - running_std[:total_steps, 1],
                    color='blue', alpha=0.2, zorder=2)

    # Set the labels and title format
    plt.xlabel(xlabel, fontsize=20, color='black')
    plt.ylabel(ylabel, fontsize=20, color='black')
    plt.gca().yaxis.get_offset_text().set_fontsize(20)
    plt.tick_params(axis='both', colors='black', direction='out', length=6, width=2, labelsize=20)

    fig.tight_layout()
    return fig


def plot_grf_2d(grf, z_pos=None, normalization = True, min_val=-1, max_val=+1, cmap='viridis', resolution= 25):
    """
    Plots a 2D slice extracted from a 3D Gaussian Random Field.

    Args:
        grf (np.ndarray): The 3D Gaussian Random Field data.
        z_pos (int): The z-position to slice for the 2D plot. Defaults to the
        middle of the z-axis.
        normalization (bool): Whether to normalize the data by 3 times
        the standard deviation. Defaults to True.
        min_val (float): Minimum value for the color scale. Defaults to -1.
        max_val (float): Maximum value for the color scale. Defaults to 1.
        cmap (str): Colormap used for plotting. Defaults to 'viridis'.
        resolution (int): Resolution of the plot figure. Defaults to 25.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # If no z-position is provided, use the middle of the z-axis
    if z_pos == None:
        z_pos = grf.shape[2]//2

    # Extract a 2D slice from the simulation box
    grf_slice = grf[:,:,z_pos]

    # Normalize the data if normalization = True
    if normalization == True:

        # Calculate the standard deviation of the data
        grf_std = np.std(grf)

        # Normalize the data by 3 times the standard deviation
        grf_slice = grf[:,:,z_pos] / ( 3*(grf_std) )

    # Create a new figure
    fig = plt.figure(figsize=(resolution, resolution))

    # Plot the 2D slice using the specified colormap and color scale
    cax = plt.imshow(grf_slice, cmap=cmap, clim=[min_val, max_val])
    plt.grid(False)
    plt.tick_params(axis='both', colors='black', direction='out', length=32, width=5, labelsize=60)
    plt.xlabel(r'$x/x_0$', fontsize=72, color='black')
    plt.ylabel(r'$y/y_0$', fontsize=72, color='black')

    plt.tight_layout()

    return fig


def logplot_grf_2d(grf, z_pos=None, normalization = True,  min_val=0.01, max_val=+1, cmap='inferno', resolution= 25):
    """
    Plots a 2D slice extracted from a 3D Gaussian Random Field with a color
    scheme on a logarithmic scale.

    Args:
        grf (np.ndarray): The 3D Gaussian Random Field data.
        z_pos (int): The z-position to slice for the 2D plot. Defaults to the
        middle of the z-axis.
        normalization (bool): Whether to normalize the data by 3 times the
        standard deviation. Defaults to True.
        min_val (float): Minimum value for the logarithmic color scale.
        Defaults to 0.01.
        max_val (float): Maximum value for the logarithmic color scale.
        Defaults to 1.
        cmap (str): Colormap used for plotting. Defaults to 'inferno'.
        resolution (int): Resolution of the plot figure. Defaults to 25.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """

    # If no z-position is provided, use the middle of the z-axis
    if z_pos == None:
        z_pos = grf.shape[2]//2

    # Extract a 2D slice from the simulation box
    grf_slice = grf[:,:,z_pos]

    # Normalize the data if normalization = True
    if normalization == True:

        # Calculate the standard deviation of the data
        grf_std = np.std(grf)

        # Normalize the data by 3 times the standard deviation
        grf_slice = grf[:,:,z_pos] / ( 3*(grf_std) )


    # Create a new figure
    fig = plt.figure(figsize=(resolution, resolution))

    # Plot the 2D slice using the specified colormap and logarithmic color scale
    cax = plt.imshow(grf_slice, cmap=cmap, norm=LogNorm(vmin=min_val, vmax=max_val))
    plt.grid(False)
    plt.tick_params(axis='both', colors='black', direction='out', length=32, width=5, labelsize=60)
    plt.xlabel(r'$x/x_0$', fontsize=72, color='black')
    plt.ylabel(r'$y/y_0$', fontsize=72, color='black')

    plt.tight_layout()
    return fig

def stats_overview(grf):
    """
    Computes and plots statistic summary for the Gaussian Random Field data.

    Args:
        grf (np.ndarray): The 3D Gaussian Random Field data.

    Returns:
        tuple: The figure object containing the plot and the
        computed statistic summary
        (standard deviation, mean, skewness, and kurtosis).
    """

    # Flatten the GRF to a 1D array
    all_points = grf.flatten()

    # Free memories
    del grf

    # Calculate statistic summary
    grf_std = np.std(all_points)
    grf_mean = np.mean(all_points)
    grf_skew = skew(all_points)
    grf_kurt = kurtosis(all_points)

    # Plot a histogram of the Gaussian Random Field data
    plt.hist(all_points, bins=256)

    # Create a string summarizing the statistics
    textstr = (f'Std. Dev. = {grf_std:.4f}\n'
            f'Mean = {grf_mean:.4f}\n'
            f'Skewness = {grf_skew:.4f}\n'
            f'Kurtosis = {grf_kurt:.4f}')

    # Add text to the plot
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title('Random Field Statistics', fontsize=16)
    plt.xlabel('Field amplitude', fontsize=16)
    plt.ylabel('Number of data points', fontsize=16)

    return plt.gcf(), grf_std, grf_mean, grf_skew, grf_kurt
