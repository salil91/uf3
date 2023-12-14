"""
This module provides functions to generate UF3 lammps potential files and provides 
help to write lammps input files for using UF3 potentials
"""

from pathlib import Path

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core import Element

from uf3.data import composition
from uf3.regression import least_squares


def generate_uf3_pot_files(model, pot_dir, overwrite=True):
    """
    Creates and writes UF3 lammps potential files.

    Args:
        model: UF3 model object
        pot_dir: Directory where the potential files will be written
        overwrite: If True, overwrites the existing files in the pot_dir

    Returns:
        List of potential files written
    """
    if not isinstance(pot_dir, Path):
        pot_dir = Path(pot_dir)
    pot_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    #TODO: Use f-strings
    for interaction in model.bspline_config.chemical_system.interactions_map[2]:
        key = '_'.join(interaction)
        files[key] = "#UF3 POT\n"
        if model.bspline_config.knot_strategy == 'linear':
            files[key] += "2B %i %i uk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)
        else:
            files[key] += "2B %i %i nk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)

        files[key] += str(model.bspline_config.r_max_map[interaction]) + " " + \
                str(len(model.bspline_config.knots_map[interaction]))+"\n"

        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.bspline_config.knots_map[interaction]]) + "\n"

        files[key] += str(model.bspline_config.get_interaction_partitions()[0][interaction]) \
                + "\n"

        start_index = model.bspline_config.get_interaction_partitions()[1][interaction]
        length = model.bspline_config.get_interaction_partitions()[0][interaction]
        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.coefficients[start_index:start_index + length]]) + "\n"

        files[key] += "#"

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            key = '_'.join(interaction)
            files[key] = "#UF3 POT\n"
            if model.bspline_config.knot_strategy == 'linear':
                files[key] += "3B %i %i uk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)
            else:
                files[key] += "3B %i %i nk\n"%(model.bspline_config.leading_trim,model.bspline_config.trailing_trim)

            files[key] += str(model.bspline_config.r_max_map[interaction][2]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][1]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][0]) + " "

            files[key] += str(len(model.bspline_config.knots_map[interaction][2])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][1])) + " " + str(len(model.bspline_config.knots_map[interaction][0])) + "\n"
            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][2]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][1]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][0]]) + "\n"

            solutions = least_squares.arrange_coefficients(model.coefficients, \
                    model.bspline_config)

            decompressed = model.bspline_config.decompress_3B( \
                    solutions[(interaction[0], interaction[1],interaction[2])], \
                    (interaction[0], interaction[1],interaction[2]))

            files[key] += str(decompressed.shape[0]) + " " \
                    + str(decompressed.shape[1]) + " " \
                    + str(decompressed.shape[2]) + "\n"

            for i in range(decompressed.shape[0]):
                for j in range(decompressed.shape[1]):
                    files[key] += ' '.join(map(str, decompressed[i,j]))
                    files[key] += "\n"
                    
            files[key] += "#"

    pot_files = []
    for k, v in files.items():
        pot_file = pot_dir / k
        pot_files.append(pot_file)
        if pot_file.is_file() and not overwrite:
            print(f"{pot_file} already exists. Skipping writing {k}")
            continue
        with open(pot_file, "w") as f:
            f.write(v)

    print("\n***Add the following line to the lammps input script***\n")
    pot_lines = get_lammps_input_file_lines(model, pot_files)
    for line in pot_lines:
        print(line, end="")
    
    return pot_files


def get_lammps_input_file_lines(model, pot_files):
    """
    Prints the lines to be added to the lammps input file for using the UF3 potential
    
    Args:
        model: UF3 model object
        pot_files: List of potential files

    Returns:
        List of lines to be added to the lammps input file
    """

    degree = model.bspline_config.degree
    element_list = model.bspline_config.chemical_system.element_list
    element_map = {element: idx for idx, element in enumerate(element_list, start=1)}
    num_elements = len(element_list)

    pot_lines = [f"pair_style uf3 {degree} {num_elements}\n"]
    for pot_file in pot_files:
        if not isinstance(pot_file, Path):
            pot_file = Path(pot_file)
        stem = pot_file.stem
        elements = stem.split("_")
        coeffs = ""
        for element in elements:
            coeffs += f"{element_map[element]} "
        coeff_str = f"pair_coeff {coeffs}{pot_file}\n"
        pot_lines.append(coeff_str)

    return pot_lines
