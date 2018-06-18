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
def sample_ising(
    spins, num_cycles, temperature, strength=1, num_thermalization_steps=0
):
    num_spins = len(spins)
    num_cycles *= num_spins

    energy_arr = np.zeros(num_cycles)
    energy = initial_energy(spins, strength)

    for i in range(num_cycles):
        ix = np.random.randint(num_spins)

        left = spins[ix - 1] if ix > 0 else spins[num_spins - 1]
        right = spins[ix + 1] if ix < (num_spins - 1) else spins[0]

        delta_energy = 2 * strength * spins[ix] * (left + right)

        if np.random.random() <= np.exp(-delta_energy / temperature):
            spins[ix] *= -1.0
            energy += delta_energy

        energy_arr[i] = energy

    assert num_thermalization_steps < energy_arr.size

    energy_arr = energy_arr[num_thermalization_steps:]

    return np.sum(energy_arr) / energy_arr.size
