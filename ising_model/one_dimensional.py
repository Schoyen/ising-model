import numpy as np
import numba


@numba.njit(cache=True)
def initial_energy(spins, strength):
    energy = 0
    num_spins = len(spins)

    energy += spins[0] * spins[num_spins - 1]

    for i in range(1, num_spins):
        energy += spins[i] * spins[i - 1]

    return -strength * energy


@numba.njit(cache=True)
def sample_ising(spins, num_cycles, temperature, strength=1):
    energy = initial_energy(spins, strength)
    num_spins = len(spins)
    norm = 1.0 / float(num_cycles)


    for cycle in range(num_cycles):
        for i in range(num_spins):
            ix = np.random.randint(num_spins)
            ix_high = (ix + 1) if ix < (num_spins - 1) else 0
            ix_low = (ix - 1) if ix > 1 else (num_spins - 1)

            delta_energy = 2 * spins[ix] * (spins[ix_high] + spins[ix_low])

        if np.random.random() <= np.exp(-delta_energy / temperature):
            spins[ix] *= -1.0
            energy += delta_energy


    return energy * norm
