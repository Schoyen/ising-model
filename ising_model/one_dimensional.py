import numpy as np
import numba

MAX_DIFF = 2
STEP_DIFF = 4


@numba.njit(cache=True)
def initialize_energy(spins, strength):
    energy = 0
    num_spins = len(spins)

    energy -= spins[0] * spins[num_spins - 1]

    for i in range(1, num_spins):
        energy -= spins[i] * spins[i - 1]

    return energy / num_spins


@numba.njit(cache=True)
def sample_ising(spins, num_cycles, temperature, strength=1):
    energy = initialize_energy(spins, strength)
    num_spins = len(spins)
    norm = 1.0 / float(num_cycles * num_spins)

    energy_diff = np.exp(
        -strength * np.arange(-MAX_DIFF, MAX_DIFF + 1, STEP_DIFF) / temperature
    )

    for cycle in num_cycles:
        for i in range(num_spins):
            ix = np.random.randint(num_spins)

            delta_energy = (
                2
                * spins[ix]
                * (spins[(ix - 1) % num_spins] + spins[(ix + 1) % num_spins])
            )

            diff_index = int((delta_energy + MAX_DIFF) / float(STEP_DIFF))

            if np.random.random() <= energy_diff[diff_index]:
                spins[ix] *= -1.0
                energy += delta_energy

    return energy * norm
