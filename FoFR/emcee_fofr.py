"""
Script for running an MCMC on the FoFR code, comparing to some mock stuff.
"""

from collections import defaultdict
import subprocess

import numpy as np
import yaml


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
    return groups


def set_params(
    b_gal: float,
    r_gal: float,
    eb: float,
    er: float,
    mag_den_scale: float,
    delta_contrast: float,
    detla_rad: float,
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
        "detla_rad": detla_rad,
        "delta_r": delta_r,
    }

    # Write the parameters back to the YAML file
    with open("parameters.yml", "w", encoding="utf-8") as file:
        yaml.dump(params, file, sort_keys=False)


def run_fofr():
    """
    runs the trystan_run.r script.
    """
    subprocess.call(["/usr/local/bin/Rscript", "trystan_run.R"])


if __name__ == "__main__":
    test_params = [0.04, 22, 0, 0, 0.7, 8, 1.4, 11]
    run_fofr()
    groups = build_group_cat("GAMA_FoFR_run.dat")
    set_params(*test_params)
    run_fofr()
    groups_two = build_group_cat('GAMA_FoFR_run.dat')
