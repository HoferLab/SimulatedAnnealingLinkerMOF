import numpy as np

from mace.calculators import mace_mp

from simulated_annealing import pq_reader, thermocycle, optimize


def main():
    # seed
    np.random.seed(42)
    # calculator
    calculator = mace_mp(model='medium',
                         dispersion=True,
                         default_dtype="float64",
                         device='cuda')

    ################### Input###################
    # first cycle
    temp_first = 5e3
    cool_first = 0.995

    # cycle
    temp_cycle = 2e2
    cool_cycle = 0.99
    cycles = 15

    # final
    temp_final_first = 1
    temp_final_cycle = 1

    # for snu78_60.xyz
    name = 'snu70_60.xyz'
    nlinker = 24
    nlinker_atoms = 20

    ################### Input ###################

    system = pq_reader(name)

    system.calc = calculator

    converged = optimize(system)
    if not converged:
        raise ValueError('Initial optimization did not converge')

    # first cycle
    system_final = thermocycle(
        system,
        nlinker,
        nlinker_atoms,
        T=temp_first,
        Tf=temp_final_first,
        update_cooling_rate=cool_first,
        cycle_number=0,
    )

    # next cycles
    for i in range(cycles):
        system_final.calc = calculator
        system_final = thermocycle(
            system_final,
            nlinker,
            nlinker_atoms,
            T=temp_cycle,
            Tf=temp_final_cycle,
            update_cooling_rate=cool_cycle,
            cycle_number=i + 1,
        )


if __name__ == '__main__':
    main()
