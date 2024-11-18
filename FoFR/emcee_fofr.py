"""
Script for running an MCMC on the FoFR code, comparing to some mock stuff.
"""

from collections import defaultdict
import subprocess

import numpy as np
import pandas as pd
import yaml

from cost import calculate_s_total


# define a function that will run the R code through python.


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
    df = pd.read_csv(infile, sep='\s+')
    group_ids = np.array(df[id_column])

    unique_ids, count = np.unique(group_ids, return_counts=True)
    cut = np.where(count>2)
    actual_groups = unique_ids[cut]
    groups = []
    for group in actual_groups:
        groups.append(np.where(group_ids == group)[0] + 1)
    return convert_listarr_listset(groups)


def set_params(
    b_gal: float,
    r_gal: float,
    eb: float,
    er: float,
    mag_den_scale: float,
    delta_contrast: float,
    delta_rad: float,
    delta_r: float,
) -> None:
    """
    Updates the parameters.yml file with the given values
    """
    # Define the new parameters in a dictionary
    params = {
        "b_gal": b_gal,
        "r_gal": r_gal,
        "Eb": eb,
        "Er": er,
        "mag_den_scale": mag_den_scale,
        "delta_contrast": delta_contrast,
        "detla_rad": delta_rad,
        "delta_r": delta_r,
    }

    # Write the parameters back to the YAML file
    with open("parameters.yml", "w", encoding="utf-8") as file:
        yaml.dump(params, file, sort_keys=False)


def convert_listarr_listset(array: list[np.ndarray]) -> list[set]:
    """
    Converts an array/list of array into a list of sets. This is important because this is the format
    that is needed in the cost function.
    """
    val = [set(arr) for arr in array]
    return val

def _run_fofr():
    """
    runs the trystan_run.r script.
    """
    subprocess.call(["/usr/local/bin/Rscript", "trystan_run.R"])


if __name__ == "__main__":
    test_params = [0.04, 22, 0, 0, 0.7, 8., 1.4, 11.]
    set_params(*test_params)
    run_fofr()
    groups = build_group_cat('GAMA_FoFR_run.dat')
    mock_groups = get_mock_groups('deep_field.dat', 'group_id_unique')
    score = calculate_s_total(groups, mock_groups)
