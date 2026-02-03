# SimulatedAnnealingLinkerMOF

*Tackling Orientational Isomerism in Metal–Organic Frameworks Comprising Low-Symmetry Linker Molecules via Simulated Annealing Featuring a Neural Network Potential*

## Concept

This repository provides a Monte Carlo (MC)-based simulated annealing tool, interfaced with the neural network potential [MACE-MP-0a](https://github.com/ACEsuit/mace-foundations), for (globally) optimizing metal–organic framework (MOF) structures with regard to the collective orientation of low-symmetry linker molecules. 

Exemplary MOFs included in this repository:
 - SNU-70
 - UiO-66(Zr)-NH<sub>2</sub>

The simulated annealing code reads MOF structures from `.xyz` files in the [PQ](https://github.com/MolarVerse/PQ) format and uses MC-based simulated annealing to optimize the orientation of organic linkers within the MOF while applying a machine learning potential (MACE-MP-0a) for energy and force calculations in each step attempt. Multiple recurring MC cooling cycles are performed. The overall goal is to find low-energy configurations close to or at the global minimum of the considered MOF, in the context of orientational isomerism.

## Installation

Clone this repository and navigate into the newly installed directory:

```sh
git clone git@github.com:HoferLab/SimulatedAnnealingLinkerMOF.git
cd SimulatedAnnealingLinkerMOF
```

If necessary, install all required Python packages as listed in `requirements.txt`, e.g., by creating a virtual environment.

## File Structure

- `snu70.py`: Script for SNU-70 optimization.
- `uio66-nh2.py`: Script for UiO-66(Zr)-NH<sub>2</sub> optimization.
- `simulated_annealing/sim_annealing.py`: Main module of the simulated annealing routine.
- `data/`: Contains the exemplary `.xyz` input files for SNU-70 and UiO-66(Zr)-NH<sub>2</sub> in the PQ format.

## Usage

Make sure that all Python packages have been pre-installed as listed in `requirements.txt`.
The input `.xyz` file must list all linker molecules at the beginning, before any other structural components.

### For SNU-70

To optimize the SNU-70 MOF structure, run:

```sh
python snu70.py 
```

This uses the input file `data/snu70_60.xyz`.

### For UiO-66(Zr)-NH<sub>2</sub>

To optimize the UiO-66(Zr)-NH<sub>2</sub> MOF structure, run:

```sh
python uio66-nh2.py
```

This uses the input file `data/amino-uio66.xyz`.

## Output Files

- `traj_annealing.traj`: Collection of accepted structures including, e.g., potential energy values.
- `orientations.npy`: Array of linker orientations (2-bit binary representation for nodal and axial orientation, respectively).
- `annealing.log`: Log file of the optimization process which lists the energies of all MC steps (successful & unsuccessful)
- `final_structure.xyz`: Optimized structures after each performed MC cooling cycle.

## License

This project is licensed under the MIT License.