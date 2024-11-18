"""
Defining the cost function which is used for optimizing to a mock catalog
"""

import numpy as np


# Functions for the 'Purity' calculation. (Equations 12 - 14)
def purity_product_sets(group_1: set, group_2: set) -> float:
    """
    Calculates the purity product for two given groups.

    Example: [1, 2, 3] vs [1, 2] and [3, 4]. In the first case the pp would be (2/3) * (2/2) whilst
    in the second case, pp = (1/3) * (1/2)
    """
    overlap = len(group_1 & group_2)
    return (overlap / len(group_1)) * (overlap / len(group_2))


def calculate_purity(group_1: set, groups_2: list[set]) -> float:
    """
    calculting the purity for a given group 1 whin compared to a list of groups (groups_2)
    """
    purity_products = [purity_product_sets(group_1, group_2) for group_2 in groups_2]
    return max(purity_products)


def calculate_q(groups_1: list[set], groups_2: list[set]) -> float:
    """
    Full Q_fof/mock algorithm (generalized equations 12 and 13 from robotham + 2011)
    """
    group_lengths = np.array([len(group) for group in groups_1])
    purities = np.array([calculate_purity(group, groups_2) for group in groups_1])
    total_group_galaxies = len(set.union(*groups_2))
    return np.sum(group_lengths * purities) / total_group_galaxies


def calculate_full_q_value(fof_groups: list[set], mock_groups: list[set]) -> float:
    """
    Main function to be used in the cost function analaysis. Equations 12-14 in Robotham+2011
    """
    q_fof = calculate_q(fof_groups, mock_groups)
    q_mock = calculate_q(mock_groups, fof_groups)
    q_total = q_fof * q_mock
    return q_total


# Functions for the global grouping efficiency Equations 9 - 11.
def count_bijective_groups(list_1: list[set], list_2: list[set]) -> int:
    """
    Counts the number of bijective groups between two lists of sets.
    A bijective group is defined as one where at least 50% of its members are shared
    with another group in both directions.

    Args:
        list_1: A list of sets representing the first collection of groups.
        list_2: A list of sets representing the second collection of groups.

    Returns:
        int: The number of bijective groups.
    """
    bijective_count = 0

    for group_1 in list_1:
        for group_2 in list_2:
            overlap = len(group_1 & group_2)

            # Calculate percentage overlap for both groups
            perc_overlap_1 = overlap / len(group_1)
            perc_overlap_2 = overlap / len(group_2)

            # If both have at least 50% overlap, it's a bijective group
            if perc_overlap_1 >= 0.5 and perc_overlap_2 >= 0.5:
                bijective_count += 1

    return bijective_count


def calculate_full_e_score(fof_groups: list[set], mock_groups: list[set]) -> float:
    """
    Calculates the full global grouping statistic (Eqs. 9-11) in Robotham+2011
    """
    number_of_bijective_groups = count_bijective_groups(fof_groups, mock_groups)
    number_of_fof_groups = len(fof_groups)
    number_of_mock_groups = len(mock_groups)
    e_fof = number_of_bijective_groups / number_of_fof_groups
    e_mock = number_of_bijective_groups / number_of_mock_groups
    e_tot = e_fof * e_mock
    return e_tot


def calculate_s_total(fof_groups: list[set], mock_groups: list[set]) -> float:
    """
    This is the main function to be used in conjunction with group catalog comparisons to mocks.
    Calculates the combined e_total and Q_total values. Eq. 15 in Robotham+2011
    """
    e_tot = calculate_full_e_score(fof_groups, mock_groups)
    q_tot = calculate_full_q_value(fof_groups, mock_groups)
    s_tot = e_tot * q_tot
    return s_tot
