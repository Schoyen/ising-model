import numpy as np
import numba


@numba.njit(cache=True)
def initial_energy(spins, strength):
    energy = 0
    num_spins = len(spins)

    for i in range(num_spins):
        for j in range(num_spins):
            left = spins[i - 1, j] if i > 0 else spins[num_spins - 1, j]
            above = spins[i, j - 1] if j > 0 else spins[i, num_spins - 1]

            energy -= strength * spins[i, j] * (left + above)

    return energy


@numba.njit(cache=True)
def sample_ising(spins, num_cycles, temperature, strength=1):
    num_spins = len(spins)
    num_cycles *= num_spins ** 2

    energy_arr = np.zeros(num_cycles)
    energy = initial_energy(spins, strength)

    for i in range(num_cycles):
        ix = np.random.randint(num_spins)
        iy = np.random.randint(num_spins)

        left = spins[ix - 1, iy] if ix > 0 else spins[num_spins - 1, iy]
        right = spins[ix + 1, iy] if ix < (num_spins - 1) else spins[0, iy]

        above = spins[ix, iy - 1] if iy > 0 else spins[ix, num_spins - 1]
        below = spins[ix, iy + 1] if iy < (num_spins - 1) else spins[ix, 0]

        delta_energy = (
            2 * strength * spins[ix, iy] * (left + right + above + below)
        )

        if np.random.random() <= np.exp(-delta_energy / temperature):
            spins[ix, iy] *= -1.0
            energy += delta_energy

        energy_arr[i] = energy

    return np.sum(energy_arr) / energy_arr.size
