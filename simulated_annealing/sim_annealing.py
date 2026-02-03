import numpy as np
from io import TextIOBase as TextIO

from ase.io import write, Trajectory
from ase.atoms import Atoms
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from ase.units import kB


def pq_reader(filename: str) -> Atoms:
    """
    Read a .xyz file in the PQ format and return an Atoms object.

    Args:
        filename (str): path to the .xyz file

    Returns:
        Atoms: Atoms object
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    natoms, a, b, c, alpha, beta, gamma = [float(i) for i in lines[0].split()]

    natoms = int(natoms)

    positions = np.array([])

    element = ""
    for line in lines[2:]:
        split_line = line.split()
        element += split_line[0]
        x, y, z = [float(i) for i in split_line[1:]]
        positions = np.append(positions, [x, y, z])

    atoms = Atoms(element, positions=positions.reshape(natoms, 3))
    atoms.set_cell([a, b, c, alpha, beta, gamma])
    atoms.set_pbc([True, True, True])

    return atoms


def optimize(atoms: Atoms, fmax: float = 0.1, steps: int = 200) -> bool:
    """
    Optimize the structure of the atoms object.

    Args:
        atoms (Atoms): Atoms object to be optimized
        fmax (float): Maximum force allowed on atoms. Default: 0.1
        steps (int): Maximum number of optimization steps. Default: 200

    Returns:
        bool: True if optimization converged, False otherwise
    """

    ucf = UnitCellFilter(atoms, hydrostatic_strain=True)

    opt = LBFGS(ucf, logfile="opt.log")

    opt.run(fmax=0.1, steps=200)

    return opt.converged()


def sorted_cov_matrix(atoms: Atoms) -> np.ndarray:
    """
    Calculate the sorted covariance matrix of the atoms object.

    Args:
        atoms (Atoms): Atoms object

    Returns:
        np.ndarray: Sorted covariance matrix
    """

    cov = np.cov(atoms.get_positions().T)

    eigval, eigvec = np.linalg.eig(cov)

    eigvec = eigvec[:, np.argsort(eigval)]

    return eigvec


def rotate(
    atoms: Atoms,
    linker_carbon_atoms: Atoms,
    vector_of_rotation: np.ndarray,
    angle: int = 180,
) -> None:
    """
    Rotate the atoms object by a given angle around a given vector.

    Args:
        atoms (Atoms): Atoms object to be rotated
        linker_carbon_atoms (Atoms): Atoms object of linker carbon atoms
        vector_of_rotation (np.ndarray): Vector of rotation
        angle (int): Angle of rotation. Default: 180

    Returns:
        None
    """

    com = linker_carbon_atoms.get_center_of_mass()

    atoms.rotate(angle, vector_of_rotation, center=com)

    return None


def update_orientation(
    orientation: np.ndarray,
    linker: int,
    vector: int,
) -> None:
    """
    Update the orientation of the linker.

    Args:
        orientation (np.ndarray): Orientation of linkers
        linker (int): Linker index
        vector (int): Vector of rotation

    Raises:
        ValueError: If vector is not in range 0-2

    Returns:
        None
    """

    # update orientations
    if vector == 0:
        orientation[linker] ^= 0b11
    elif vector == 1:
        orientation[linker] ^= 0b10
    elif vector == 2:
        orientation[linker] ^= 0b01
    else:
        raise ValueError("Vector not in range 0-2")

    return None


def write_acceptance_ratio(
    f: TextIO,
    energy: float,
    T: float,
    ratio: float,
) -> None:
    """
    Write the acceptance ratio to the log file.

    Args:
        f (TextIO): Log file
        energy (float): Energy of the system
        T (float): Temperature
        ratio (float): Acceptance ratio

    Returns:
        None
    """

    f.write(f"\t\t{energy:.2f}\t{T:.2f}\t\t{ratio:.3f}\n")
    f.flush()

    return None


def thermocycle(
    system: Atoms,
    nlinker: int,
    nlinker_atoms: int,
    T: float = 1e3,
    Tf: float = 1e-1,
    update_cooling_rate: float = 0.99,
    cycle_number: int = 0,
    trajectory_name: str = "traj_annealing.traj",
) -> None:
    """
    Perform a thermocycle on the atoms object.

    Args:
        system (Atoms): Atoms object
        nlinker (int): Number of linkers
        nlinker_atoms (int): Number of linker atoms
        T (float): Initial temperature. Default: 1e3
        Tf (float): Final temperature. Default: 1e-1
        update_cooling_rate (float): Cooling rate. Default: 0.99
        cycle_number (int): Number of cycles. Default: 0
        trajectory_name (str): Name of the trajectory file. Default: 'traj_annealing.traj'

    Returns:
        None
    """

    # open log file
    f = open("annealing.log", "a")

    f.write(f"Cycle #{cycle_number}\tEnergy\t\tTemperature\tAcceptance Ratio\n")

    # Output orientation of linkers
    try:
        orientations = np.load("orientations.npy").tolist()
        orientation = orientations[-1]
    except:
        # binary list of binary 00 - descriptor of linker orientations
        orientation = np.array([0b00] * nlinker)

        orientations = [orientation.copy()]

    energy = system.get_potential_energy()

    energies = [energy]

    # trajectory
    trajectory = Trajectory(trajectory_name, "a", system)

    # ratio of accepted steps
    accepted, steps = 0, 1

    if cycle_number == 0:
        f.write(f"\t\t{energy:.2f}\tInitial structure!\n")
        f.flush()
        trajectory.write(system)

    # final structure
    system_min = system.copy()

    while T > Tf:
        # select linker
        linker = np.random.randint(0, nlinker)

        # linker indices
        linker_start = linker * nlinker_atoms
        linker_end = (linker + 1) * nlinker_atoms

        # select which rotation to do
        vector = np.random.randint(0, 3)

        # select linker system
        linker_atoms = system[linker_start:linker_end]
        linker_carbon_atoms = linker_atoms.copy()

        # remove non-carbon atoms
        del linker_carbon_atoms[
            [atom.index for atom in linker_carbon_atoms if atom.symbol != "C"]
        ]

        # save old positions
        old_positions = system.positions.copy()

        # rotate linker by 180 degrees around selected vector
        vector_of_rotation = sorted_cov_matrix(linker_carbon_atoms)[:, vector]
        rotate(linker_atoms, linker_carbon_atoms, vector_of_rotation, angle=180)

        # update system positions with new linker positions
        system.positions[linker_start:linker_end] = linker_atoms.positions

        # optimize system and check if optimization converged
        converged = optimize(system)
        if not converged:
            system.positions = old_positions
            continue

        # get new energy of the system
        energy = system.get_potential_energy()

        # Metropolis criterion
        delta_energy = energy - energies[-1]
        if np.min([1, np.exp(-delta_energy * 1 / (kB * T))]) >= np.random.rand():
            # accept new positions

            # update orientations
            update_orientation(orientation, linker, vector)

            # save new orientation
            orientations.append(orientation.copy())

            # save new structure
            trajectory.write(system)

            # update acceptance ratio
            accepted += 1

            write_acceptance_ratio(f, energy, T, accepted / steps)

            # update minimum energy structure
            if energy < min(energies):
                system_min = system.copy()

            # update energy
            energies.append(energy)
            np.save("orientations.npy", orientations)

        else:
            # don't accept new positions

            # revert to old positions
            system.positions = old_positions

            write_acceptance_ratio(f, energy, T, accepted / steps)

        # decrease temperature
        T *= update_cooling_rate
        steps += 1

    # write final structure
    system_min.calc = system.calc
    system_min.get_potential_energy()
    write("final_structure.xyz", system_min, format="extxyz", append=True)
    f.close()

    return system_min
