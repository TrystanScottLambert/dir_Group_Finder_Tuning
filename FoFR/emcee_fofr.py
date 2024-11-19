"""
Script for running an MCMC on the FoFR code, comparing to some mock stuff.
"""

import os
from collections import defaultdict
import subprocess
import datetime

import numpy as np
import pylab as plt
import pandas as pd
import yaml
import emcee
import corner

from cost import calculate_s_total
from make_R import make_r

def convert_listarr_listset(array: list[np.ndarray]) -> list[set]:
    """
    Converts an array/list of array into a list of sets. This is important because this is the 
    typing that is needed in the cost function.
    """
    val = [set(arr) for arr in array]
    return val

def build_group_cat(infile: str) -> list[np.ndarray]:
    """
    Reads in the output file from the R script and converts it into a list of arrays that can be
    read by the cost function.
    """
    gal_id, group_id = np.loadtxt(infile, unpack=True)
    groups_dict = defaultdict(list)
    for gal, grp in zip(gal_id, group_id):
        groups_dict[grp].append(gal)

    groups = [np.array(groups_dict[g_id]) for g_id in groups_dict]
    return convert_listarr_listset(groups)

def get_mock_groups(infile: str, id_column: str) -> list[np.ndarray]:
    """
    Creates the list of groups from the mock catalaog that are needed for the cost function.
    """
    df = pd.read_csv(infile, sep=' ')
    group_ids = np.array(df[id_column])

    unique_ids, count = np.unique(group_ids, return_counts=True)
    cut = np.where(count>2)
    actual_groups = unique_ids[cut]
    groups = []
    for group in actual_groups:
        groups.append(np.where(group_ids == group)[0] + 1)
    return convert_listarr_listset(groups)

def run_fofr(r_script_path, input_file, output_file):
    """
    Runs the trystan_run.R script with specified input and output files.
    """
    subprocess.run(
            ["Rscript", r_script_path, input_file, output_file],
            check=True
        )


def log_probability(params, param_bounds, mock_groups):
    """
    Log-probability function for MCMC with Python-R integration.
    """
    # Check parameter bounds
    for param, bounds in zip(params, param_bounds):
        if not (bounds[0] <= param <= bounds[1]):
            return -np.inf

    unique_name = str(abs(hash(str(datetime.datetime.now()) + str(np.random.random(100000)))))
    unique_script_name = unique_name + '.R'
    unique_yaml_name = unique_name + '.yml'
    unique_dat_name = unique_name + '.dat'

    # Create unique temporary files for input and output
    with open(unique_yaml_name, 'w', encoding='utf-8') as input_file, open(unique_dat_name, 'w', encoding='utf-8') as output_file:

        # Write parameters to the input YAML file
        params_dict = {
            "b_gal": float(params[0]),
            "r_gal": float(params[1]),
            "Eb": float(params[2]),
            "Er": float(params[3]),
            "mag_den_scale": float(params[4]),
            "delta_contrast": float(params[5]),
            "delta_rad": float(params[6]),
            "delta_r": float(params[7]),
        }
        yaml.dump(params_dict, input_file, sort_keys=False)
        input_file_path = input_file.name
        output_file_path = output_file.name



    make_r(unique_script_name)
    try:
        # Run the R script
        run_fofr(unique_script_name, input_file_path, output_file_path)

        # Read groups from the R output file
        groups = build_group_cat(output_file_path)

        # Compute the score
        s_total = calculate_s_total(groups, mock_groups)

        # Return the negative score (for MCMC maximization)
        return -s_total

    except subprocess.CalledProcessError as e:
        print(f"Error in R script execution {e}")
        return -np.inf

    finally:
        # Clean up temporary files
        os.remove(input_file_path)
        os.remove(output_file_path)
        os.remove(unique_script_name)

def run_mcmc(mock_groups, param_bounds, n_walkers=50, n_steps=1000):
    """
    Runs MCMC to find the best parameters.

    Parameters:
    mock_groups (list of list): The real groups from the mock catalog.
    param_bounds (list of tuples): Bounds for each parameter.
    r_script_path (str): Path to the R script file.
    n_walkers (int): Number of MCMC walkers.
    n_steps (int): Number of MCMC steps.

    Returns:
    emcee.EnsembleSampler: The MCMC sampler with the results.
    """
    n_params = len(param_bounds)

    # Initial positions of walkers: Randomly sample within the bounds
    initial_positions = [
        [np.random.uniform(low, high) for low, high in param_bounds]
        for _ in range(n_walkers)
    ]

    # Set up the sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        log_probability,
        args=(param_bounds, mock_groups)  # Pass additional arguments
    )

    # Run the MCMC sampling
    sampler.run_mcmc(initial_positions, n_steps, progress=True)

    return sampler

def main():
    """
    Main function for scoping.
    """
    #theta = [0.04, 22, 0, 0, 0.7, 8., 1.4, 11.]
    bounds = [(0.0, 0.1), (18, 25), (0, 1), (0, 1), (0, 2), (5, 12), (0, 3), (8, 15)]
    mock_groups = get_mock_groups('deep_field.dat', 'group_id_unique')
    print('Running the MCMC sampling...')

    sampler = run_mcmc(mock_groups, bounds, n_steps=50, n_walkers=20)
    samples = sampler.get_chain(flat=True)
    best_params = np.median(samples, axis=0)
    print('Best-fit parameters: ', best_params)

    # Plot the results
    fig = corner.corner(samples, truths=best_params)
    plt.show()


if __name__ == "__main__":
    main()
