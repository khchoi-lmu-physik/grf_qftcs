# @title

# Import packages and libraries

from scipy.special import hankel2
from scipy.stats import skew, kurtosis

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import time as ti

# @title 1.1 Random field simulations

class grf_sim:

    def __init__(self, pixel=2**9, mean=0, std_dev=1):
        """
        Initializes the grf_sim class with default or provided parameters.

        Upon initialization, Gaussian white noise is automatically sampled
        and its Fourier Transform is computed. This is the basis
        for generating the Gaussian Random Fields in curved spacetimes.

        Args:
            pixel (int): Resolution of the field (number of pixels per dimension).
            mean (float): Mean of the Gaussian white noise. Default to 0.
            std_dev (float): Standard deviation of the Gaussian white noise.
                             Default to 1.
        """
        self.pixel = pixel
        self.mean = mean
        self.std_dev = std_dev
        self.fft_white_noise, self.k_norm = self.grf_initialization()

    def grf_initialization(self):
        """
        Initializes the Gaussian Random Fields simulation by generating white
        noise and computing its Fast Fourier Transform.

        This function is automatically compiled upon calling the grf_sim class.

        Returns:
            fft_white_noise (cp.ndarray): Fourier-transformed white noise.
            k_norm (cp.ndarray): Norm of the momentum space grid.
        """
        # Sampling Gaussian white noise
        white_noise = cp.random.normal(self.mean, self.std_dev, (self.pixel, self.pixel, self.pixel))

        # Compute the FFT of the Gaussian white noise
        fft_white_noise = cp.fft.fftn(white_noise).astype(cp.complex128)
        # Free up memeory
        del white_noise

        # Generate FFT momentum frequencies for each dimension
        kx = cp.fft.fftfreq(self.pixel)*self.pixel
        ky = cp.fft.fftfreq(self.pixel)*self.pixel
        kz = cp.fft.fftfreq(self.pixel)*self.pixel

        # Create a 3D grid of momentum vectors
        kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz)
        # Free up memory
        del kx, ky, kz

        # Compute the norm of the momentum vector k
        k_norm =  cp.sqrt( kx_grid**2 + ky_grid**2 + kz_grid**2 ) # k = sqrt( kx^2 + ky^2  + kz^2 )

        # Regularize divergence at k=0, for any power spectrum of spectral index less than.
        k_norm[0,0,0] = cp.inf
        # Free up memories
        del kx_grid, ky_grid, kz_grid

        # Return FFT white noise and the norm of momentum vector k
        return fft_white_noise, k_norm

    def generate_grf_desitter(self, time=1, amplitude=1/10, n_index=1.5):
        """
        Generates a Gaussian Random Field based on the de Sitter power spectrum.

        Args:
            time (float): Time variable at which the Gaussian Random Field is
            simulated. Default to 1.
            amplitude (float): Amplitude scaling factor for the field.
            Default to 1/10.
            n_index (float): Spectral index for the power spectrum.
            Default to 3/2.

        Returns:
            gaussian_random_field (cp.ndarray): Real part of the generated
            Gaussian Random Field.
        """

        # Convert momentum norm and amplitude to NumPy arrays for CPU operations
        k_norm = cp.asnumpy(self.k_norm)
        # Amplitude of the field
        amplitude = cp.asnumpy(amplitude)


        """
        Calculate the Fourier amplitude using the Klein-Gordon solution
        in De-Sitter Space.

        The generation of Gaussian Random Fields in DeSitter space is not
        optimized here due to the lack of Hankel function in CuPy.
        The author still needs to implement the Hankel function using CUDA here.

        """
        fourier_amplitudes_sqrt =  np.sqrt(np.pi) * 0.5 *  amplitude   *  (time**(3/2)) * hankel2(n_index, time * ( (2 * np.pi/self.pixel)* k_norm) )
        # Free up memories
        del k_norm

        # Regularize divergence at k=0 by setting the amplitude to zero
        fourier_amplitudes_sqrt[0,0,0] = 0

        # Convert the Fourier amplitudes to a CuPy array and specifying data type with complex128.
        fourier_amplitudes_sqrt = cp.asarray(fourier_amplitudes_sqrt).astype(cp.complex128)

        # Take the absolute value of the Fourier amplitudes and specifying data type with complex128
        fourier_amplitudes_sqrt = cp.abs(fourier_amplitudes_sqrt).astype(cp.complex128)

        # Multiply with Fourier amplitude with FFT white noise
        fourier_amplitudes_sqrt *= self.fft_white_noise

        # Perform inverse FFT to get the real-space Gaussian Random Fields
        gaussian_random_field = cp.fft.ifftn(fourier_amplitudes_sqrt)

        # Free up memories
        del fourier_amplitudes_sqrt

        # Return the real part of the Gaussian Random Fields
        return gaussian_random_field.real


    def generate_grf_rdu(self, time=1, amplitude=1):
        """
        Generates a Gaussian Random Field based on the power spectrum of
        radiation-dominated universe (RDU).

        Args:
            time (float): Time variable at which the Guassian random fields are
            simulated. Default to 1.
            amplitude (float): Amplitude scaling factor for the GRF.
             Default to 1.

        Returns:
            gaussian_random_field (cp.ndarray): Real part of the generated GRF.
        """

        # Calculate the power spectrum of a scalar field in the RDU
        power_spectrum = amplitude * (1/2) * (time ** (-2)) * ( (2 * cp.pi / self.pixel) * self.k_norm) ** (-1)

        # Compute the square root of the power spectrum, and specify data type as complex128
        fourier_amplitudes_sqrt = cp.sqrt(power_spectrum).astype(cp.complex128)
        # Free up memory
        del power_spectrum

        # Multiply with the FFT of the white noise
        fourier_amplitudes_sqrt *= self.fft_white_noise

        # Perform inverse FFT to get the real-space Guassian Random Fields
        gaussian_random_field = cp.fft.ifftn(fourier_amplitudes_sqrt)
        # Free up memory
        del fourier_amplitudes_sqrt

        # Return the real part of the Gaussian Random Fields
        return gaussian_random_field.real

    def run_simulation_desitter(self,  start_time, stop_time, time_step, run_id=0,  amplitude = 1/10, n_index=1.5, extra_time_range = [], save_grf_time = [], plot_bool = False ):
        """
        Runs a Gaussian Random Field simulation over a specified time range
        wtih the De Sitter spacetimes as the background geomerty.

        Args:
            start_time (float): Starting time of the simulation.
            No deafult value.
            stop_time (float): Ending time of the simulation. No deafult value.
            time_step (float): Time step between simulations. No deafult value.
            run_id (int): Identifier for the multi-simulation run.
            Defaults to 0.
            amplitude (float): Amplitude scaling factor for the quantum field.
            Defaults to 1/10.
            n_index (float): Spectral index for the power spectrum.
            Defaults to 3/2.
            extra_time_range (list): Additional times outside the specified
            time range to include for the simulation. Defaults to None.
            save_grf_time (list): Times at which the Gaussian Random Fields
            data should be saved. Defaults to None.
            plot_bool (bool): Whether to plot results during the simulation.
            Defaults to False.

        Saves:
            Gaussian Random Field data (`grf`) at specified time points
            if `plot_bool` is True. The data is saved in .npy format
            The filename is identified by the time and resolution in the
            format of `grf_t={time:.4f}_pixel={self.pixel}.npy`.
            Statistical measures collected during the simulation
            (e.g., standard deviation, mean, skewness, kurtosis)
            are automatically saved to disk at the end of the simulation.
        """

        t0 = ti.time()

         # Calculate the number of steps based on time range and step size
        num_steps = int( (stop_time - start_time) / time_step) + 1

        # Create a timeline for the simulation
        time_line = np.linspace(start_time, stop_time, num_steps)

        # The timeline includes the specified time range and additional times
        time_line = np.concatenate( (time_line,  np.array(extra_time_range) ) )

        # Initialize statistics dictionary to store statistics
        statistics  = {'std': [], 'mean': [], 'skew': [], 'kurt': [], 'max_val': [], 'over_2std': []}

        for time in time_line:

            t1 = ti.time()

            # Generate the Gaussian Rrandom Field for the current time step
            grf = self.generate_grf_desitter(time=time, amplitude=amplitude, n_index=n_index)

            """
            This line computes the statistical measures for the Gaussian
            Random Fields:

            1. Standard deviation (grf_std).
            2. Mean (grf_mean).
            3. Skewness (grf_skew): Measures the asymmetry of the Gaussian
            Random Field distribution.
            4. Kurtosis (grf_kurt): Measures the tailness of the Gaussian
            Random Field distribution.
            5. `stats_overview` generates and returns a histogram (`stat_fig`)
            of the flattened Gaussian random field data.
            """
            # Gather statistics of the Gaussian Random Field
            stat_fig, grf_std, grf_mean, grf_skew, grf_kurt = self.stats_overview(grf)

            # Flatten and analyze the data
            """
            The following operations flatten and analyze the Gaussian Random
            Field data for:
            1. The maximum value
            2. Finding all data points that exceed 2 Std.
            3. Calculates the percentage of data points that exceed 2 Std.
            """
            # Flatten the data
            grf_abs_flatten = cp.abs(grf).flatten()

            # Find the maximum of the data
            grf_max_val = cp.max(grf_abs_flatten).get()

            # Identify all the data points that exceed 2 Std.
            samples_over_2std = grf_abs_flatten[grf_abs_flatten > (2 * grf_std)]

            # Calculate the prcentage of data points that exceed 2 Std.
            percent_over_2std = (samples_over_2std.size / grf_abs_flatten.size) * 100

            # Store the statistics in the dictionary
            for key, value in zip(statistics.keys(), [grf_std, grf_mean, grf_skew, grf_kurt, grf_max_val, percent_over_2std]):
                statistics[key].append([time, value])

            # Optionally save Gaussian Random Field in (.npy) format, and print computational time.
            if run_id == 0 and plot_bool == True and round(time,4) in save_grf_time:

                # Save Gaussian random field data in .npy format.
                cp.save(f'grf_t={time:.4f}_pixel={self.pixel}.npy', grf)

                tf = ti.time() - t1

                # Print computational time
                print(f't= {time:.4f}: Computational time is: {tf:.2f} seconds')
                plt.close('all')

        # Save statistics dictionary
        self.save_statistics(statistics, run_id)

        # Print total computational time
        tf = ti.time() - t0
        print(f'Run: {run_id+1}: Computational time for this runs is: {tf:.2f} seconds')
        plt.close('all')


    def run_simulation_rdu(self,  start_time, stop_time, time_step, run_id=0,  amplitude = 1, extra_time_range = [], save_grf_time = [], plot_bool = False ):
        """
        Runs a Gaussian Random Field (GRF) simulation over a specified time
        range in the radiation-dominated universe (RDU).

        Args:
            start_time (float): Starting time of the simulation.
            No default value.
            stop_time (float): Ending time of the simulation. No default value.
            time_step (float): Time step between simulations. No default value.
            run_id (int): Identifier for the multi-simulation run. Defaults to 0.
            amplitude (float): Amplitude scaling factor for the quantum field.
            Defaults to 1.
            extra_time_range (list): Additional times outside the specified
            time range to include in the simulation. Defaults to None.
            save_grf_time (list): Specific times at which the GRF data
            should be saved. Defaults to None.
            plot_bool (bool): Whether to plot results during the simulation.
            Defaults to False.

        Saves:
            Gaussian Random Field data (`grf`) at specified time points
            if `plot_bool` is True. The data is saved in .npy format.
            The filename is identified by the time and resolution in the
            format of `grf_t={time:.4f}_pixel={self.pixel}.npy`.
            Statistical measures collected during the simulation
            (e.g., standard deviation, mean, skewness, kurtosis)
            are automatically saved to disk at the end of the simulation.
        """
        t0 = ti.time()

        # Calculate the number of simulation steps based on the time range and time step
        num_steps = int( (stop_time - start_time) / time_step) + 1

        # Create a timeline for the simulation
        time_line = np.linspace(start_time, stop_time, num_steps)

        # The timeline includes the specified time range and additional times
        time_line = np.concatenate( (time_line,  np.array(extra_time_range) ) )

        # Initialize a dictionary to store statistics
        statistics  = {'std': [], 'mean': [], 'skew': [], 'kurt': [], 'max_val': [], 'over_2std': []}

        for time in time_line:

            t1 = ti.time()

            # Generate the Gaussian Random Field for the current time step
            grf = self.generate_grf_rdu(time=time, amplitude=amplitude)

            """
            This line computes the statistical measures for the Gaussian Random Fields:

            1. Standard deviation (grf_std).
            2. Mean (grf_mean).
            3. Skewness (grf_skew): Measures the asymmetry of the Gaussian
            Random Field distribution.
            4. Kurtosis (grf_kurt): Measures the tailness of the Gaussian
            Random Field distribution.
            5. `stats_overview` generates and returns a histogram (`stat_fig`)
            of the flattened Gaussian random field data.
            """
            # Gather statistics of the Gaussian Random Field
            stat_fig, grf_std, grf_mean, grf_skew, grf_kurt = self.stats_overview(grf)

            # Flatten and analyze the data
            """
            The following operations flatten and analyze the Gaussian Random
            Field data for:
            1. The maximum value
            2. Finding all data points that exceed 2 Std.
            3. Calculates the percentage of data points that exceed 2 Std.
            """
            # Flatten the data
            grf_abs_flatten = cp.abs(grf).flatten()

            # Find the maximum of the data
            grf_max_val = cp.max(grf_abs_flatten).get()

            # Identify all the data points that exceed 2 Std.
            samples_over_2std = grf_abs_flatten[grf_abs_flatten > (2 * grf_std)]

            # Calculate the percentage of data points that exceed 2 Std.
            percent_over_2std = (samples_over_2std.size / grf_abs_flatten.size) * 100

            # Store the statistics in the dictionary
            for key, value in zip(statistics.keys(), [grf_std, grf_mean, grf_skew, grf_kurt, grf_max_val, percent_over_2std]):
                statistics[key].append([time, value])

            # Optionally save Gaussian Random Field in (.npy) format, and print computational time.
            if run_id == 0 and plot_bool == True and round(time,4) in save_grf_time:

                # Save Gaussian random field data in .npy format.
                cp.save(f'grf_t={time:.4f}_pixel={self.pixel}.npy', grf)

                tf = ti.time() - t1

                # Print computational time
                print(f't= {time:.4f}: Computational time is: {tf:.2f} seconds')
                plt.close('all')

        # Save statistics dictionary
        self.save_statistics(statistics, run_id)

        tf = ti.time() - t0

        # Print total computational time
        print(f'Run: {run_id+1}: Computational time for this runs is: {tf:.2f} seconds')
        plt.close('all')


    def stats_overview(self, grf):
        """
        Computes and returns key statistical measures for the given Gaussian
        Random Field data.

        Args:
            grf (cp.ndarray): The Gaussian random field (GRF) data.

        Returns:
            stat_fig (matplotlib.figure.Figure): A figure object showing the
            histogram of the GRF.
            grf_std (float): Standard deviation of the GRF data.
            grf_mean (float): Mean value of the GRF data.
            grf_skew (float): Skewness of the GRF data.
            grf_kurt (float): Kurtosis of the GRF data.
        """

        # Convert the Gaussian Random field data into a NumPy array and flatten for analysis
        all_points = cp.asnumpy(grf.flatten())

        # Free memories
        del grf

        # Compute statistical measures

        # Calculate standard deviation, mean, skewness, and kurtosis
        grf_std = np.std(all_points)
        grf_mean = np.mean(all_points)
        grf_skew = skew(all_points)
        grf_kurt = kurtosis(all_points)

        # Plot a histogram of the GRF values with 256 bins
        plt.hist(all_points, bins=256)

        # Create a string for of statistics summaries
        textstr = (f'Std. Dev. = {grf_std:.4f}\n'
                f'Mean = {grf_mean:.4f}\n'
                f'Skewness = {grf_skew:.4f}\n'
                f'Kurtosis = {grf_kurt:.4f}')


        # Add statistical measures as a text box to the plot
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Set plot titles and x-y labels
        plt.title('Random Field Statistics', fontsize=16)
        plt.xlabel('Field amplitude', fontsize=16)
        plt.ylabel('Number of data points', fontsize=16)

        return plt.gcf(), grf_std, grf_mean, grf_skew, grf_kurt


    def save_statistics(self, statistics, run_id):
        """
        Saves the statistical measures of the Guassian Radnom Field (GRF)
        to .npy files.

        Args:
            statistics (dict): Dictionary containing statistical measures
            for each time step.
            run_id (int): Identifier for the different simulation run.

        Saves:
            `percent_over_2std_array_run_{run_id+1}.npy`:
                Contains the percentage of data points in the GRF that
                exceed 2 Std. for each time step.

            `max_val_array_run_{run_id+1}.npy`:
                Stores the maximum absolute value in the GRF for each time step.

            `std_array_run_{run_id+1}.npy`:
                Contains the standard deviation of the GRF for each time step.

            `mean_array_run_{run_id+1}.npy`:
                Stores the mean value of the GRF for each time step.

            `skew_array_run_{run_id+1}.npy`:
                Contains the skewness of the GRF for each time step.

            `kurt_array_run_{run_id+1}.npy`:
                Stores the kurtosis of the GRF for each time step.
        """
        np.save(f'percent_over_2std_array_run_{run_id+1}.npy', np.array(statistics['over_2std']))
        np.save(f'max_val_array_run_{run_id+1}.npy', np.array(statistics['max_val']) )
        np.save(f'std_array_run_{run_id+1}.npy', np.array(statistics['std']))
        np.save(f'mean_array_run_{run_id+1}.npy', np.array(statistics['mean']))
        np.save(f'skew_array_run_{run_id+1}.npy', np.array(statistics['skew']))
        np.save(f'kurt_array_run_{run_id+1}.npy', np.array(statistics['kurt']))
