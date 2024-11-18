"""
Functions to write the correct columns for Aarons code. 
The catalog must be in ra, dec, redshift, and magnitude.
"""

import pandas as pd


def create_essential_catalog(
    file_name: str,
    outfile: str,
    ra_name: str,
    dec_name: str,
    redshift_name: str,
    mag_name: str,
) -> None:
    """
    Opens the file and creates a new version with only ra, dec, z, mag.

    Parameters:
    file_name (str): The path to the input file (CSV).
    outfile (str): The path to save the output file.
    ra_name (str): The column name for Right Ascension.
    dec_name (str): The column name for Declination.
    redshift_name (str): The column name for Redshift.
    mag_name (str): The column name for Magnitude.
    """

    df = pd.read_csv(file_name, sep="\s+")

    selected_columns = [ra_name, dec_name, redshift_name, mag_name]
    selected_columns = [col for col in selected_columns if col in df.columns] 
    essential_df = df[selected_columns]
    essential_df.to_csv(outfile, sep = ' ', index=False)

if __name__ == '__main__':
    infile = 'deep_field.dat'
    outfile = 'deep_field_group_finding.dat'
    col_names = ['ra', 'dec', 'zobs', 'r_VST_ap']
    create_essential_catalog(infile, outfile, *col_names)
